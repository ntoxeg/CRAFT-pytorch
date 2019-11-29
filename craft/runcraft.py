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


def run_net(
    net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None
):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
        image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5
    )
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    y, feature = net(x)

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
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = cvt2HeatmapImg(render_img)

    # if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


def run_on_img(imgbytes):
    # load net
    net = CRAFT()  # initialize
    cuda = torch.cuda.is_available()
    refine = False
    refiner_model = None
    trained_model_weights = "craft_mlt_25k.pth"

    print("Loading weights from checkpoint (" + trained_model_weights + ")")
    if cuda:
        net.load_state_dict(copyStateDict(torch.load(trained_model_weights)))
    else:
        net.load_state_dict(
            copyStateDict(torch.load(trained_model_weights, map_location="cpu"))
        )

    if cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if refine:
        from refinenet import RefineNet

        refine_net = RefineNet()
        print("Loading weights of refiner from checkpoint (" + refiner_model + ")")
        if cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(
                copyStateDict(torch.load(refiner_model, map_location="cpu"))
            )

        refine_net.eval()
        args.poly = True

    t = time.time()

    # load data
    # for k, image_path in enumerate(image_list):
    # print(
    #     "Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path),
    #     end="\r",
    # )
    image = loadImage(imgbytes)

    bboxes, polys, score_text = run_net(
        net, image, 0.7, 0.4, 0.4, cuda, False, refine_net
    )

    # save score text
    # filename, file_ext = os.path.splitext(os.path.basename(img))
    # mask_file = result_folder + "/res_" + filename + "_mask.jpg"
    # cv2.imwrite(mask_file, score_text)

    # saveResult(img, image[:, :, ::-1], polys, dirname=result_folder)

    print("elapsed time : {}s".format(time.time() - t))
    return bboxes, polys, score_text
