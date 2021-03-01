# -*- coding: utf-8 -*-
import json
import time
import threading
import requests
from demo import *
from flask import Flask, request, Response

app = Flask(__name__)
global jsonResponse0
global jsonResponse1
global get_vid_path

def main_func_0(get_vid_path, get_pic_path):
    print('helloooo')
    url = 'http://127.0.0.1:5001/test0'
    # url = 'http://192.168.99.1:5001/test0'
    response1 = jsonResponse1.copy()
    time0 = time.time()
    results = main_func(get_vid_path, get_pic_path)
    print(time.time() - time0)

    response1["results"] = results
    print(response1)
    #requests.post(url, json=response1)


@app.route('/test', methods=['GET','POST'])
def dault():

    response0 = jsonResponse0.copy()
    my_json = request.get_json()
    get_vid_path = my_json.get("video_path")
    get_pic_path = my_json.get("imgs_path")

    #check
    try:
        if os.path.exists(get_vid_path):
            print('***')
            flag = True

            end = get_vid_path.split('.')[-1]
            print(end)
            if end != 'mp4':
                response0['code'] = 30000
                response0['msg'] = '视频格式不匹配'
                return Response(json.dumps(response0), mimetype='application/json')

            if flag == True:
                response0['code'] = 10000
                response0['msg'] = '传输成功,请等待。。。！'
                mthread = threading.Thread(target=main_func_0, args=(get_vid_path, get_pic_path))
                mthread.start()
    except:
        response0['code'] = 30000
        response0['msg'] = '路径不存在'
        return Response(json.dumps(response0), mimetype='application/json')

    return Response(json.dumps(response0), mimetype='application/json')

if __name__ == '__main__':
    jsonResponse0 = {"code": 10000, "msg": ""}
    jsonResponse1 = {"results":""}

    # app.run(host='192.168.99.1', port=5000)
    app.run(host='0.0.0.0', port=5000)