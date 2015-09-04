# -*- coding: utf-8 -*-
"""
Created on Mon May 04 12:56:39 2015

@author: Gonçalo
"""

import cv2
import numpy as np

class RoiSet:
    def __init__(self, rois, offset=(0,0), scale=(1,1),
                 dtype=np.float64, flipxy=False):
        self.rois = [[(x*scale[0]+offset[0],
                       y*scale[1]+offset[1])
                     for x,y in roi]
                     for roi in rois]
        if flipxy:
            self.rois = [[(y,x) for x,y in roi] for roi in self.rois]
        self.area = [cv2.contourArea(np.array(roi,dtype=np.float32))
                     for roi in self.rois]
        self.center = [np.mean(roi,axis=0,dtype=dtype) for roi in self.rois]
        self.min = [np.min(roi,axis=0) for roi in self.rois]
        self.max = [np.max(roi,axis=0) for roi in self.rois]
        
    def __repr__(self):
        return str.format("RoiSet({0})",self.rois)
        
    def __str__(self):
        return str(self.rois)