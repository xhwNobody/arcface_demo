import os
import cv2
import mxnet as mx
import numpy as np
import mxnet.ndarray as nd
from threading import Lock
import time
_file = 'detector2.py'
cur_path = os.getcwd()

#fd_model = os.path.join(cur_path, 'model','R50')
fd_model = 'C:\\Users\\test\\Desktop\\preform\\Retinaface_insightface\\retinaface_checkpoint\\retinaface_checkpoint'

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def anchors_plane(height, width, stride, base_anchors):
    """
    Parameters
    ----------
    height: height of plane
    width:  width of plane
    stride: stride ot the original image
    anchors_base: (A, 4) a base set of anchors
    Returns
    -------
    all_anchors: (height, width, A, 4) ndarray of anchors spreading over the plane
    """
    A = base_anchors.shape[0]
    all_anchors = np.zeros((height, width, A, 4), dtype=np.float32)
    for iw in range(width):
        sw = iw * stride
        for ih in range(height):
            sh = ih * stride
            for k in range(A):
                all_anchors[ih, iw, k, 0] = base_anchors[k, 0] + sw
                all_anchors[ih, iw, k, 1] = base_anchors[k, 1] + sh
                all_anchors[ih, iw, k, 2] = base_anchors[k, 2] + sw
                all_anchors[ih, iw, k, 3] = base_anchors[k, 3] + sh
    return all_anchors

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6), stride=16):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    return anchors

def generate_anchors_fpn(cfg):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
    RPN_FEAT_STRIDE = []
    for k in cfg:
      RPN_FEAT_STRIDE.append( int(k) )
    RPN_FEAT_STRIDE = sorted(RPN_FEAT_STRIDE, reverse=True)
    anchors = []
    for k in RPN_FEAT_STRIDE:
      v = cfg[str(k)]
      bs = v['BASE_SIZE']
      __ratios = np.array(v['RATIOS'])
      __scales = np.array(v['SCALES'])
      stride = int(k)
      #print('anchors_fpn', bs, __ratios, __scales, file=sys.stderr)
      r = generate_anchors(bs, __ratios, __scales, stride)
      #print('anchors_fpn', r.shape, file=sys.stderr)
      anchors.append(r)

    return anchors

def clip_pad(tensor, pad_shape):
    """
    Clip boxes of the pad area.
    :param tensor: [n, c, H, W]
    :param pad_shape: [h, w]
    :return: [n, c, h, w]
    """
    H, W = tensor.shape[2:]
    h, w = pad_shape

    if h < H or w < W:
      tensor = tensor[:, :, :h, :w].copy()

    return tensor

def bbox_pred(boxes, box_deltas):
    """
    Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    :param boxes: !important [N 4]
    :param box_deltas: [N, 4 * num_classes]
    :return: [N 4 * num_classes]
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

    dx = box_deltas[:, 0:1]
    dy = box_deltas[:, 1:2]
    dw = box_deltas[:, 2:3]
    dh = box_deltas[:, 3:4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
    # y1
    pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
    # x2
    pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
    # y2
    pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

    if box_deltas.shape[1]>4:
      pred_boxes[:,4:] = box_deltas[:,4:]

    return pred_boxes

def landmark_pred(boxes, landmark_deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, landmark_deltas.shape[1], landmark_deltas.shape[2]))
    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
    pred = landmark_deltas.copy()
    for i in range(5):
        pred[:,i,0] = landmark_deltas[:,i,0]*widths + ctr_x
        pred[:,i,1] = landmark_deltas[:,i,1]*heights + ctr_y
    return pred

def _umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.

    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.

    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.

    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573

    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return T

face_src = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]], dtype=np.float32 )

# landmark is prediction; src is template
def norm_crop(img, landmark, image_size=112):
    assert landmark.shape==(5,2)
    M = _umeyama(landmark, face_src, True)[0:2,:]
    warped = cv2.warpAffine(img, M, (image_size, image_size))
    return warped

# from gpu_nms import gpu_nms as _nms
# from cpu_nms import cpu_nms as _nms
def _nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

class FaceDetector:
    def __init__(self, minsize=46, fuzzy=100, nms=0.4, fix_size=None):
        self.fuzzy = fuzzy
        self.image_size = (720, 1280)
        if fix_size is not None:
            self.image_size = fix_size
        self.minsize = int(minsize * self.image_size[0] / 1080)
        self.maxsize = int(810 * self.image_size[0] / 1080)
        sym, arg_params, aux_params = mx.model.load_checkpoint(fd_model, 0)
        ctx = mx.gpu(0)
        # ctx = mx.cpu()
        model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
        data_shape = (1,3)+self.image_size
        model.bind(data_shapes=[('data', data_shape)])
        model.set_params(arg_params, aux_params)
        # #warmup
        self.model = model
        self.nms_threshold = nms
        self.landmark_std = 1
        self._feat_stride_fpn = [32, 16, 8]
        self.anchor_cfg = {
                '32': {'SCALES': (32,16), 'BASE_SIZE': 16, 'RATIOS': (1.,), 'ALLOWED_BORDER': 9999},
                '16': {'SCALES': (8,4), 'BASE_SIZE': 16, 'RATIOS': (1.,), 'ALLOWED_BORDER': 9999},
                '8': {'SCALES': (2,1), 'BASE_SIZE': 16, 'RATIOS': (1.,), 'ALLOWED_BORDER': 9999},
                }

        self.fpn_keys = []
        for s in self._feat_stride_fpn:
            self.fpn_keys.append('stride%s'%s)
        self._anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn(cfg=self.anchor_cfg)))
        for k in self._anchors_fpn:
            v = self._anchors_fpn[k].astype(np.float32)
            self._anchors_fpn[k] = v
        self.anchor_plane_cache = {}
        self._num_anchors = dict(zip(self.fpn_keys, [anchors.shape[0] for anchors in self._anchors_fpn.values()]))
        self.detector(np.zeros((self.image_size[0],self.image_size[1],3),np.uint8))

    def detector(self, image, threshold=0.6):
        proposals_list = []
        scores_list = []
        landmarks_list = []

        im_tensor = np.transpose(image, (2,0,1))
        im_tensor = np.expand_dims(im_tensor, axis=0)

        data = nd.array(im_tensor.copy())
        db = mx.io.DataBatch(data=(data,), provide_data=[('data', data.shape)])
        self.model.forward(db, is_train=False)
        net_out = self.model.get_outputs()
        for _idx,s in enumerate(self._feat_stride_fpn):
            _key = 'stride%s'%s
            stride = int(s)
            idx = _idx*3

            scores = net_out[idx].asnumpy()
            scores = scores[:, self._num_anchors['stride%s'%s]:, :, :]
            bbox_deltas = net_out[idx+1].asnumpy()
            height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]
            A = self._num_anchors['stride%s'%s]
            K = height * width
            key = (height, width, stride)
            if key in self.anchor_plane_cache:
                anchors = self.anchor_plane_cache[key]
            else:
                anchors_fpn = self._anchors_fpn['stride%s'%s]
                anchors = anchors_plane(height, width, stride, anchors_fpn)
                anchors = anchors.reshape((K * A, 4))
                if len(self.anchor_plane_cache)<100:
                    self.anchor_plane_cache[key] = anchors

            scores = clip_pad(scores, (height, width))
            scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
            scores_ravel = scores.ravel()
            order = np.where(scores_ravel>=threshold)[0]
            scores = scores[order]
            scores_list.append(scores)
            anchors = anchors[order, :]

            bbox_deltas = clip_pad(bbox_deltas, (height, width))
            bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))
            bbox_pred_len = bbox_deltas.shape[3]//A
            bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
            bbox_deltas = bbox_deltas[order, :]
            proposals = bbox_pred(anchors, bbox_deltas)
            proposals_list.append(proposals)

            landmark_deltas = net_out[idx+2].asnumpy()
            landmark_deltas = clip_pad(landmark_deltas, (height, width))
            landmark_pred_len = landmark_deltas.shape[1]//A
            landmark_deltas = landmark_deltas.transpose((0, 2, 3, 1)).reshape((-1, 5, landmark_pred_len//5))
            landmark_deltas *= self.landmark_std
            landmark_deltas = landmark_deltas[order, :]
            landmarks = landmark_pred(anchors, landmark_deltas)
            landmarks_list.append(landmarks)

        det = None
        landmarks = None
        proposals = np.vstack(proposals_list)
        if proposals.shape[0]!=0:
            scores = np.vstack(scores_list)
            scores_ravel = scores.ravel()
            order = scores_ravel.argsort()[::-1]
            proposals = proposals[order, :]
            scores = scores[order]
            landmarks = np.vstack(landmarks_list)
            landmarks = landmarks[order].astype(np.float32, copy=False)
            pre_det = np.hstack((proposals[:,0:4], scores)).astype(np.float32, copy=False)
            keep = _nms(pre_det, self.nms_threshold)
            det = pre_det[keep]
            landmarks = landmarks[keep]
        return det, landmarks


    def fixImage(self, image):
        scale = 1.0
        x1 = y1 = 0
        if image.shape[0:2] != self.image_size:
            scx = self.image_size[1] / image.shape[1]
            scy = self.image_size[0] / image.shape[0]
            scale = min(scx,scy)
            if scale < 1.0 :
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            else:
                scale = 1

            if image.shape[0:2] != self.image_size:
                imageBg = np.zeros((self.image_size[0],self.image_size[1],3),np.uint8)
                x1 = (imageBg.shape[1]-image.shape[1])//2
                y1 = (imageBg.shape[0]-image.shape[0])//2
                imageBg[y1:y1+image.shape[0], x1:x1+image.shape[1]]=image
                image = imageBg

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, scale, (x1,y1)

    def getAll(self, image0, threshold=0.85, limit=6):
        H, W, _= image0.shape
        image, scale, org = self.fixImage(image0)
        bboxes, landmarks = self.detector(image, threshold)
        if bboxes is None:
            return None

        sort = (bboxes[:,2]-bboxes[:,0]).argsort()[::-1]
        bboxes = bboxes[sort]
        landmarks = landmarks[sort]

        faceImgs  = []
        faceRects = []
        invalRects= []
        for i in range(bboxes.shape[0]):
            bboxO = bboxes[i, 0:4].astype('int')
            bbox = [int((bboxO[0]-org[0])/scale),
                    int((bboxO[1]-org[1])/scale),
                    int((bboxO[2]-org[0])/scale),
                    int((bboxO[3]-org[1])/scale)]
            score = bboxes[i,4]
            landmark = landmarks[i]
            if score >0.9 and limit>0:
                adj = max(int((bbox[2] - bbox[0]) * 0.1), 10)
                cbox = [bbox[0]-adj, bbox[1]-adj, W-bbox[2]-adj, H-bbox[3]-adj]
                if cbox[0]>0 and cbox[1]>0 and cbox[2]>0 and cbox[3]>0:
                    x1 = landmark[2][0]-landmark[0][0]
                    x2 = landmark[1][0]-landmark[0][0]
                    rcs = x1/x2 if x2!=0 else 0
                    x1 = landmark[2][0]>landmark[3][0] and landmark[2][0]<landmark[4][0]
                    y1 = (landmark[0][1]+landmark[1][1])/2.0
                    y2 = (landmark[3][1]+landmark[4][1])/2.0 - y1
                    tcs = (landmark[2][1] - y1)/y2 if y2!=0 else 0
                    if rcs >0.35 and rcs <0.65 and tcs>0.3 and tcs <0.7 and x1 :
                        landmark0 = (landmark - org) / scale
                        cropImg = norm_crop(image0, landmark = landmark0)
                        cropImg = cv2.cvtColor(cropImg, cv2.COLOR_BGR2RGB)
                        cropGray= cv2.cvtColor(cropImg, cv2.COLOR_RGB2GRAY)
                        if cv2.Laplacian(cropGray, cv2.CV_16S).var()  > self.fuzzy:
                            faceImgs.append(cropImg)
                            faceRects.append(bbox)
                            limit -= 1
                            continue

            invalRects.append(bbox)
        return faceImgs, faceRects, invalRects

    def getOne(self, image, threshold=0.85):
            image, _, _= self.fixImage(image)
            bboxes, landmarks = self.detector(image, threshold)
            if bboxes is None:
                return None
            h, w, _ = image.shape
            match = 0
            minsw = w
            for i in range(len(bboxes)):
                sw = abs(w - bboxes[i][0] -bboxes[i][2])
                if (sw < minsw):
                    minsw = sw
                    match = i
            return norm_crop(image, landmark = landmarks[match])

    def getAllRaw(self, image, threshold=0.6):
        image, scale, org = self.fixImage(image)
        bboxes, landmarks = self.detector(image, threshold)
        if bboxes is None:
            return None, None
        faceImgs  = []
        faceRects = []
        for i in range(bboxes.shape[0]):
            bboxO = bboxes[i, 0:4].astype('int')
            # bboxO[0] = bboxO[0] - 20
            # bboxO[1] = bboxO[1] - 20
            # bboxO[2] = bboxO[2] + 20
            # bboxO[3] = bboxO[3] + 20
            # bboxO = [bboxO[0] - 20, bboxO[1] - 20, bboxO[2] + 20, bboxO[3] + 20]
            bbox = [int((bboxO[0] - org[0]) / scale),
                    int((bboxO[1] - org[1]) / scale),
                    int((bboxO[2] - org[0]) / scale),
                    int((bboxO[3] - org[1]) / scale)]
            cropImg = norm_crop(image, landmark = landmarks[i])
            cropImg = cv2.cvtColor(cropImg, cv2.COLOR_BGR2RGB)
            faceImgs.append(cropImg)
            faceRects.append(bbox)
        return faceImgs, faceRects

if __name__ == '__main__':
    path = 'test_data\\detect_images\\'
    dir = os.listdir(path)
    rg = FaceDetector()
    for i in dir:
        filename1 = os.path.join(path, i)
        image1 = cv2.imread(filename1)
        h ,w, _ = image1.shape
        faceImgs1, faceRects1 = rg.getAllRaw(image1)
        # if faceRects1 is not None:
        #     for a, j in enumerate(faceRects1):
        #         x1, x2, y1, y2 = j[0], j[2], j[1], j[3]
        #         x1 = x1 - 20
        #         x2 = x2 + 20
        #         y1 = y1 - 20
        #         y2 = y2 + 20
        #         if x1 <0 or x2>w or y1<0 or y2>h:
        #             continue
        #         pic = image1[y1:y2, x1:x2]
        #         print('img:', i, pic.shape, x1, x2, y1, y2)
        #         cv2.imwrite(os.path.join('.\\detect_outputs\\', str(a)+'_'+i), pic)

        if faceImgs1 is not None:
            for id, img in enumerate(faceImgs1):
                cv2.imwrite(os.path.join('test_data\\detect_outputs\\', '11_' + str(id) + '_' + i), img)

    del rg
