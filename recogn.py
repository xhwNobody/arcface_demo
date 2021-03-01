# -*- coding: utf-8 -*-
import os
import threading
import numpy as np
import mxnet as mx

cpath = os.getcwd()

def load_model(ctx, prefix, epoch):
    #print('loading model', prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    sym = sym.get_internals()['fc1_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    # model.bind(data_shapes=[('data', (batch_size, 3, 112, 112))], label_shapes=[('softmax_label', (batch_size,))])
    model.bind(data_shapes=[('data', (1, 3, 112, 112))])
    model.set_params(arg_params, aux_params)
    return model

def normalize(v):
    return v / np.linalg.norm(v)

class FaceRecognition:
    def __init__(self):
        self.model = None
        self.ga_model = None
        self.ctx = mx.gpu(0)
        # self.ctx = mx.cpu(0)
        self.image_size = (112, 112)
        # self.lock = threading.Lock()
        self.model = load_model(self.ctx, os.path.join(cpath, 'arcface_checkpoint', 'arcface_checkpoint'), 0)

    def getFeature(self, rgbImage):
        aligned = np.transpose(rgbImage, (2, 0, 1))
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob, ctx=self.ctx)
        db = mx.io.DataBatch(data=(data,))
        # self.lock.acquire()
        self.model.forward(db, is_train=False)
        result = self.model.get_outputs()[0].asnumpy()
        # self.lock.release()
        return normalize(result[0])

    def getFeatures(self, rgbImages, size=2):
        outputs = []
        count = len(rgbImages)
        if count == 0: return []
        rgbImages = np.transpose(rgbImages, (0, 3, 1, 2))
        if count % size != 0:
            rgbImages = np.append(rgbImages, np.zeros((size - (count % size), 3, 112, 112)), 0)
        for i in range(0, count, size):
            data = mx.nd.array(rgbImages[i:i + size], ctx=self.ctx)
            data = mx.io.DataBatch(data=(data,))
            # self.lock.acquire()
            self.model.forward(data, is_train=False)
            result = self.model.get_outputs()[0].asnumpy()
            # self.lock.release()
            for j in range(result.shape[0]):
                outputs.append(normalize(result[j]))
        return outputs[0:count]


class GaRecognition:
    def __init__(self):
        self.model = None
        self.ga_model = None
        self.ctx = mx.gpu(0)
        # self.ctx = mx.cpu(0)
        self.lock = threading.Lock()
        self.ga_model = load_model(self.ctx, os.path.join(cpath, 'gamodel', 'model'), 0)

    def getGa(self, rgbImage):
        aligned = np.transpose(rgbImage, (2, 0, 1))
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob, ctx=self.ctx)
        db = mx.io.DataBatch(data=(data,))
        self.lock.acquire()
        self.ga_model.forward(db, is_train=False)
        result = self.ga_model.get_outputs()[0].asnumpy()
        self.lock.release()
        gender = np.argmax(result[:, 0:2])
        age = result[:, 2:202].reshape((100, 2))
        age = np.argmax(age, axis=1)
        age = int(sum(age))
        return gender, age

    def getGas(self, rgbImages, size=2):
        outputs = []
        count = len(rgbImages)
        rgbImages = np.transpose(rgbImages, (0, 3, 1, 2))
        if count % size != 0:
            rgbImages = np.append(rgbImages, np.zeros((size - (count % size), 3, 112, 112)), 0)
        for i in range(0, count, size):
            data = mx.nd.array(rgbImages[i:i + size], ctx=self.ctx)
            data = mx.io.DataBatch(data=(data,))
            self.lock.acquire()
            self.ga_model.forward(data, is_train=False)
            result = self.ga_model.get_outputs()[0].asnumpy()
            self.lock.release()
            gender = result[:, 0:2]
            age = result[:, 2:202].reshape((size, 100, 2))
            for j in range(result.shape[0]):
                outputs.append([np.argmax(gender[j]), sum(np.argmax(age[j], axis=1))])
        return outputs[0:count]


if __name__ == '__main__':
    import cv2

    specific_name = {0: 'chenhe'}
    specific_path = r'C:\\Users\\test\\Desktop\\preform\\Retinaface_insightface\\recog_moban\\'
    dir = os.listdir(specific_path)
    rg = FaceRecognition()
    specific_feas = {}
    for file in dir:
        id = file.split('.')[0]
        image = cv2.imread(specific_path + file)
        image = cv2.resize(image, (112, 112))
        image = image[:, :, ::-1]
        featrue = rg.getFeature(image)
        specific_feas[id] = featrue

    need_path = r'C:\\Users\\test\\Desktop\\preform\\Retinaface_insightface\\recog_images\\'
    need_dir = os.listdir(need_path)
    for file in need_dir:
        print(file)
        img1 = cv2.imread(need_path + file)
        image1 = cv2.resize(img1, (112, 112))
        image1 = image1[:, :, ::-1]
        pred_fea = rg.getFeature(image1)
        dists = []
        for id in specific_feas:
            dist = np.sum(np.square(specific_feas[id] - pred_fea))
            print(dist)
            dists.append(dist)

        if dists != [] and min(dists) < 1.1:
            print(specific_name[dists.index(min(dists))])
        else:
            print(None)





