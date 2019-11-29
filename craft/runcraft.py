# -*- coding: utf-8 -*-
"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

import argparse
import json
import os
import sys
import time
import zipfile
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image
from skimage import io
from torch.autograd import Variable

from .craft import CRAFT
from .craft_utils import *

# from .file_utils import *
from .imgproc import *


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


class CRAFTRunner(object):
    def __init__(self):
        self.net = CRAFT()  # initialize
        self.cuda = torch.cuda.is_available()
        refine = False
        refiner_model = None
        trained_model_weights = "craft_mlt_25k.pth"

        print("Loading weights from checkpoint (" + trained_model_weights + ")")
        if self.cuda:
            self.net.load_state_dict(copyStateDict(torch.load(trained_model_weights)))
        else:
            self.net.load_state_dict(
                copyStateDict(torch.load(trained_model_weights, map_location="cpu"))
            )

        if self.cuda:
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False

        self.net.eval()

        # LinkRefiner
        self.refine_net = None
        if refine:
            from .refinenet import Refinenet

            self.refine_net = Refinenet()
            print("Loading weights of refiner from checkpoint (" + refiner_model + ")")
            if self.cuda:
                self.refine_net.load_state_dict(
                    copyStateDict(torch.load(refiner_model))
                )
                self.refine_net = self.refine_net.cuda()
                self.refine_net = torch.nn.DataParallel(self.refine_net)
            else:
                self.refine_net.load_state_dict(
                    copyStateDict(torch.load(refiner_model, map_location="cpu"))
                )

            self.refine_net.eval()
            args.poly = True

        # load data
        # for k, image_path in enumerate(image_list):
        # print(
        #     "Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path),
        #     end="\r",
        # )

    def run_net(
        self,
        image,
        text_threshold,
        link_threshold,
        low_text,
        cuda,
        poly,
        refine_net=None,
    ):
        t = time.time()
        t0 = time.time()

        # resize
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
            image, 1280 * 2, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5
        )
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
        if cuda:
            x = x.cuda()

        # forward pass
        y, feature = self.net(x)

        # make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        # refine link
        if refine_net is not None:
            y_refiner = refine_net(y, feature)
            score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

        t0 = time.time() - t0
        t1 = time.time()

        # Post-processing
        boxes, polys = getDetBoxes(
            score_text, score_link, text_threshold, link_threshold, low_text, poly
        )

        # coordinate adjustment
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None:
                polys[k] = boxes[k]

        t1 = time.time() - t1

        # render results (optional)
        # render_img = score_text.copy()
        # render_img = np.hstack((render_img, score_link))
        # ret_score_text = cvt2HeatmapImg(render_img)
        ret_score_text = score_text

        # if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

        print("elapsed time : {}s".format(time.time() - t))
        return boxes, polys, ret_score_text

    def run_on_img(self, img):

        image = loadImage(img)

        bboxes, polys, score_text = self.run_net(
            image, 0.7, 0.4, 0.4, self.cuda, False, self.refine_net
        )

        # save score text
        # filename, file_ext = os.path.splitext(os.path.basename(img))
        # mask_file = result_folder + "/res_" + filename + "_mask.jpg"
        # cv2.imwrite(mask_file, score_text)

        # saveResult(img, image[:, :, ::-1], polys, dirname=result_folder)

        return bboxes, polys, score_text
