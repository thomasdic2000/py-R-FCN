#!/usr/bin/env python

# --------------------------------------------------------
# R-FCN
# Copyright (c) 2016 Yuwen Xiong
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__',
           'solar')
#           'aeroplane', 'bicycle', 'bird', 'boat',
#           'bottle', 'bus', 'car', 'cat', 'chair',
#           'cow', 'diningtable', 'dog', 'horse',
#           'motorbike', 'person', 'pottedplant',
#           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'ResNet-101': ('ResNet-101',
		  'solar_resnet101_rfcn_iter_40000.caffemodel'), 
#                  'resnet101_rfcn_final.caffemodel'),
        'ResNet-50': ('ResNet-50',
                  'resnet50_rfcn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    #ax.figure.savefig(out_name, bbox_inches='tight')
    #plt.close()

def save_detections(out_name, im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    ax.figure.savefig(out_name, bbox_inches='tight')
    plt.close()

def demo(net, image_name, img_path ):
  """Detect object classes in an image using pre-computed object proposals."""
    # Load the demo image
  im_file = os.path.join(img_path, image_name)
  if os.path.isfile(im_file):
      try:
          im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
          timer = Timer()
          timer.tic()
          scores, boxes = im_detect(net, im)
          timer.toc()
          print ('Detection took {:.3f}s for '
              '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
          CONF_THRESH = 0.8
          NMS_THRESH = 0.3
          output_name = os.path.join(img_path, 'output', image_name)

          for cls_ind, cls in enumerate(CLASSES[1:]):
              cls_ind += 1 # because we skipped background
              cls_boxes = boxes[:, 4:8]
              cls_scores = scores[:, cls_ind]
              dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
              keep = nms(dets, NMS_THRESH)
              dets = dets[keep, :]
        #vis_detections(im, cls, dets, thresh=CONF_THRESH)
              save_detections(output_name, im, cls, dets, thresh=CONF_THRESH)
      except:
	print("Exception occured")

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description=' R-FCN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [ResNet-101]',
                        default='ResNet-101')
    parser.add_argument('--cfg', dest='demo_cfg', help='Optional config to use',
                        default=None, type=str)
    parser.add_argument('--input', dest='input_dir', help='input directory to use',
                        default=None, type=str)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if args.demo_cfg is not None:
	cfg_file = args.demo_cfg
        cfg_from_file(cfg_file)	

    prototxt = 'models/solar_panel/ResNet-101/rfcn_end2end/test_agnostic.prototxt'
    #prototxt = os.path.join(cfg.ROOT_DIR, 'models/solar_panel',  NETS[args.demo_net][0],
    #                        'rfcn_end2end', 'test_agnostic.prototxt')
#    if args.demo_net is None:
#    	caffemodel = os.path.join(cfg.ROOT_DIR, 'output/rfcn_end2end/solar_panel', NETS[args.demo_net][1])
#    else:
    caffemodel = args.demo_net

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg']

    if args.input_dir:
        for file in os.listdir(args.input_dir):
	  output_path = os.path.join(args.input_dir, 'output')
          if not os.path.exists(output_path):
	      os.makedirs(output_path)

          out_file = os.path.join(output_path, file)
  	  if not os.path.isfile(out_file):
            if file.endswith(".jpg"):
		demo(net, file, args.input_dir)
    else:
      for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        img_path = os.path.join(cfg.DATA_DIR, 'demo')
        demo(net, im_name, img_path)

    #plt.show()
