#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import warnings
import cv2
from PIL import Image
from yolo import YOLO
from tools import generate_detections as gdet
import cPickle
import numpy as np
from deep_sort import preprocessing
from deep_sort.detection import Detection
warnings.filterwarnings('ignore')

def main(yolo):

    nms_max_overlap = 1.0
    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    video_capture = cv2.VideoCapture(0)
        
    ret, frame = video_capture.read()  # frame shape 640*480*3
    
    image = Image.fromarray(frame[...,::-1]) #bgr to rgb
    boxs = yolo.detect_image(image)
    features = encoder(frame,boxs)
    print(features)
    
    for count, feature in enumerate(features):
        #create new 2d arrays for each individual
        box_arr = [boxs[count]]
        feat_arr = [feature]
        detections = [Detection(bbox, 1.0, feat) for bbox, feat in zip(box_arr, feat_arr)]
            
        # Run non-maxima suppression. gets rid of annoying overlapping BB's that are likely the same object
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
        
        #output frame
        cv2.startWindowThread()
        cv2.namedWindow("preview")
        cv2.imshow("preview", frame)
        cv2.waitKey(2000)
        cv2.imwrite("pic"+str(count)+".jpg",frame)
        
        #write individual feature vectors to pickle files
        pickle_out = open("pickledump"+str(count)+".pickle","wb")
        cPickle.dump(feature, pickle_out)
        pickle_out.close()
        
        #read and print pickled feature vectors
        pickle_in = open("pickledump"+str(count)+".pickle","rb")
        ret = cPickle.load(pickle_in)
        print(ret)
    
    video_capture.release()
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main(YOLO())