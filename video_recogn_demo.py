# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from detector import FaceDetector
from recogn import FaceRecognition

facer_detecor = FaceDetector()
facer_recogn = FaceRecognition()

#参数设置
frame_frequency = 1
face_verification_threshold = 1.0

#计算embedding之间的距离
def cal_dist(fea1, fea2):
    return np.sum(np.square(fea1 - fea2))

def video_2(video_path, features_template_list):

    vide_capture = cv2.VideoCapture(video_path)
    frame_height = vide_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = vide_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = vide_capture.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_datou.avi', fourcc, fps, (int(frame_width), int(frame_height)))

    frame_id = 0
    while vide_capture.isOpened():
        ret, image = vide_capture.read()


        if ret:
            if frame_id % frame_frequency == 0:

                pred_ids_dict = {}
                #人脸检测
                faceImgs1, faceRects1 = facer_detecor.getAllRaw(image)

                faceImgs2 = []
                if faceImgs1 != None:
                    faceImgs2 = [img for img in faceImgs1 if img.shape[0]>0 and img.shape[1]>0]

                #人脸识别
                features = []
                if len(faceImgs2) > 0:
                    #人脸尺寸标准化

                    face_choose_resize = [cv2.resize(face, (112, 112)) for face in faceImgs2]

                    #将每个人脸转换为embedding
                    for face_choose in face_choose_resize:
                        face_choose = face_choose[:, :, ::-1]
                        feature = facer_recogn.getFeature(face_choose)
                        features.append(feature)
                    features = np.array(features)

                    #计算帧人脸与每个模板之间的距离
                    pred_dist = np.zeros([features.shape[0], len(features_template_list)]) + 2
                    for i in range(features.shape[0]):
                        for j in range(len(features_template_list)):
                            pred_dist[i][j] = cal_dist(features[i], features_template_list[j])

                    #依据最小距离确定人脸id
                    pred_sim2 = pred_dist.tolist()

                    for idt, sig in enumerate(pred_sim2):
                        if min(sig) < face_verification_threshold:
                            id = sig.index(min(sig))
                            #pred_ids.append(id)
                            pred_ids_dict[id] = idt

                if pred_ids_dict != []:
                    print(pred_ids_dict)
                    #for idx, id_ in enumerate(pred_ids):
                    for idx, id_ in pred_ids_dict.items():
                        bbox = faceRects1[id_]
                        #pred_id = pred_ids_dict[idx]
                        pred_id = idx

                        point_0 = (bbox[0], bbox[1])
                        point_1 = (bbox[2], bbox[3])
                        cv2.rectangle(image, point_0, point_1, (0,0,255),2)
                        cv2.putText(image,characterNames[pred_id], (point_0[0]-10, point_0[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                out.write(image)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    return
            frame_id += 1
        else:
            out.release()
            break


if __name__ == '__main__':

    face_verification_threshold = 1.0

    #计算出模板任务的embedding
    video_path = 'datouerzi.mp4'
    imgs_path = ['datou.jpg', 'dapang.jpg', 'damei.jpg', 'dameinv.jpg']
    characterNames = []
    fea_list = []
    for img_path in imgs_path:
        img_data = cv2.imread(img_path)
        img_data = cv2.resize(img_data, (112, 112))
        img_data = img_data[:, :, ::-1]
        fea = facer_recogn.getFeature(img_data)
        fea_list.append(fea)
        characterNames.append(os.path.split(img_path)[-1].split('.')[0])

    record_times = video_2(video_path, fea_list)
