import argparse
import logging
import sys
import time

# from tf_pose import common
from tf_pose import mycommon as common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def getOpenPose(img,resize='432x368',model='mobilenet_thin',resize_out_ratio=4.0):


    w, h = model_wh(resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

    # estimate human poses from a single image !
    image = common.read_imgfile(img, None, None)
    if image is None:
        logger.error('Image can not be read, path=%s' % img)
        sys.exit(-1)

    t = time.time()
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)
    elapsed = time.time() - t

    logger.info('inference image: %s in %.4f seconds.' % (img, elapsed))

    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    show_image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow("color",show_image)
    cv2.waitKey(0)





if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_w)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_h)

    ret, frame = cap.read()
    w, h= cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # img=frame.resize(VIDEO_W,VIDEO_H)
    img=frame

    resize=str(int(w))+'x'+str(int(h))


    # path='images/p1.jpg'
    # img=cv2.imread(path, cv2.IMREAD_COLOR)

    getOpenPose(img,resize)




    cap.release()
    cv2.destroyAllWindows()
