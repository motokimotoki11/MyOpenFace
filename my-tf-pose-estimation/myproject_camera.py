import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.myestimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

# FPSのMAXが2.5程度　PC性能に依存
SAMPLING_FREQUENCY=3
start_time=time.time()


logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def get_postures(point):
    # ０：鼻、１：心臓、２：右肩、3：右肘、４：右手首、５：左肩、６：左肘、７：左手首、８：右腰、
    # ９：右膝、10：右足首、11：左腰、12：左膝、13：左足首、14：右目、15：左目、16：右耳、17：左耳
    # POSITION=['鼻','首の付け根','右肩','右肘','右手首','左肩','左肘','左手首','右腰','右膝',
    #           '右足首','左腰','左膝','左足首','右目','左目','右耳','左耳']

    main_pos=[[0,0],[0,0],[0,0],[0,0]]

    for i,j in point.items():
        if i==0: #鼻
            main_pos[0]=j
        elif i==1: #首の付け根
            main_pos[1]=j
        elif i==14: #右目
            main_pos[2]=j
        elif i==15: #左目
            main_pos[3]=j

    return main_pos

def show_openpose(image):
    global start_time
    # cv2.putText(image,
    #             "FPS: %f" % (1.0 / (time.time() - fps_time)),
    #             (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (0, 255, 0), 2)
    cv2.putText(image,
                "TIME: %f" % (time.time()-start_time ),
                (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)
    cv2.imshow('tf-pose-estimation result', image)

def show_eye(image,r_eye,l_eye,nose):

    global start_time


    if r_eye[0]!=0:
        if nose[0]==0:
            roi_range=30
        else:
            roi_range=int(abs(nose[0]-r_eye[0])*0.7)
        #上下左右
        r_roi=[r_eye[1]-roi_range,r_eye[1]+roi_range,r_eye[0]-roi_range,r_eye[0]+roi_range]
        r_roi=[0 if i<0 else i for i in r_roi]
        cv2.rectangle(image, (r_roi[3],r_roi[0]), (r_roi[2],r_roi[1]), (255, 255, 255), 2)

    if l_eye[0]!=0:
        if nose[0]==0:
            roi_range=30
        else:
            roi_range=int(abs(nose[0]-l_eye[0])*0.7)
        #上下左右
        l_roi=[l_eye[1]-roi_range,l_eye[1]+roi_range,l_eye[0]-roi_range,l_eye[0]+roi_range]
        l_roi=[0 if i<0 else i for i in l_roi]

        cv2.rectangle(image, (l_roi[3],l_roi[0]), (l_roi[2],l_roi[1]), (255, 255, 255), 2)
    cv2.imshow('eye result', image)



def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_openpose(camera=0,resize='432x368',resize_out_ratio=4.0,model='mobilenet_thin',show_process=False,tensorrt='False'):
    #camera 0:内部カメラ float
    #resize ピクセル str
    #resize_out_ratio ヒートマップが後処理される前にサイズを変更
    #model 'cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small'こっから選ぶ　float
    #show_process デバッグ目的で、有効にすると、推論の速度が低下します。 boolean
    #tensorrt 学習済みDeep Learningモデルを、GPU上で高速に推論できるように最適化してくれるライブラリ　str

    global SAMPLING_FREQUENCY
    global start_time

    fps_time = time.time()


    logger.debug('initialization %s : %s' % (model, get_graph_path(model)))
    w, h = model_wh(resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(model), target_size=(w, h), trt_bool=str2bool(tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368), trt_bool=str2bool(tensorrt))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    while True:
        ret_val, image = cam.read()

        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)

        # openPose表示
        openpose_image= TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        show_openpose(openpose_image)

        # 目のROI etc
        # pos= TfPoseEstimator.my_positions(image, humans, imgcopy=False)
        # nose,neck,r_eye,l_eye=get_postures(pos)
        # show_eye(image,r_eye,l_eye,nose)





        while (time.time() - fps_time)< (1/SAMPLING_FREQUENCY):
            continue

        fps_time = time.time()
        if cv2.waitKey(1) == ord('q'):
            break
        # logger.debug('finished+')

    cv2.destroyAllWindows()



if __name__ == '__main__':
    get_openpose()
