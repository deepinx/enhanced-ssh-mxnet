from __future__ import print_function
import argparse
import sys
import os
import time
import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
import cv2
from essh_detector import ESSHDetector
from logger import logger
from config import config, default, generate_config
#from rcnn.tools.test_rcnn import test_rcnn
#from rcnn.tools.test_rpn import test_rpn
from rcnn.processing.bbox_transform import nonlinear_pred, clip_boxes
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.nms import gpu_nms_wrapper
from rcnn.processing.bbox_transform import bbox_overlaps
from rcnn.dataset import widerface


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Faster R-CNN network')
    # general
    parser.add_argument('--network', help='network name', default=default.network, type=str)
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset)
    parser.add_argument('--image_set', help='image_set name', default=default.test_image_set, type=str)
    parser.add_argument('--root_path', help='output data folder', default=default.root_path, type=str)
    parser.add_argument('--dataset_path', help='dataset path', default=default.dataset_path, type=str)
    parser.add_argument('--method_name', help='method name for official WIDER toolbox', default='ESSH-Pyramid', type=str)
    # testing
    parser.add_argument('--prefix', help='model to test with', default=default.e2e_prefix, type=str)
    parser.add_argument('--epoch', help='model to test with', default=0, type=int)
    parser.add_argument('--gpu', help='GPU device to test with', default=0, type=int)
    parser.add_argument('--output', help='output folder', default=os.path.join('./', 'output'), type=str)
    parser.add_argument('--pyramid', help='enable pyramid test', action='store_true')
    # rcnn
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    parser.add_argument('--thresh', help='valid detection threshold', default=0.05, type=float)
    parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
    parser.add_argument('--has_rpn', help='generate proposals on the fly', action='store_true', default=True)
    parser.add_argument('--proposal', help='can be ss for selective search or rpn', default='rpn', type=str)
    args = parser.parse_args()
    return args

detector = None
args = None

def get_boxes(roi, pyramid):
  im = cv2.imread(roi['image'])
  if not pyramid:
    target_size = 1200
    max_size = 1600
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    scales = [im_scale]
  else:
    TEST_SCALES = [500, 800, 1200, 1600]
    target_size = 800
    max_size = 1200
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    scales = [float(scale)/target_size*im_scale for scale in TEST_SCALES]
  boxes = detector.detect(im, threshold=args.thresh, scales = scales)
  return boxes


def test(args):
  print('test with', args)
  global detector
  output_folder = args.output
  if not os.path.exists(output_folder):
    os.mkdir(output_folder)
  detector = ESSHDetector(args.prefix, args.epoch, args.gpu, test_mode=True)
  imdb = eval(args.dataset)(args.image_set, args.root_path, args.dataset_path)
  roidb = imdb.gt_roidb()
  gt_overlaps = np.zeros(0)
  overall = [0.0, 0.0]
  gt_max = np.array( (0.0, 0.0) )
  num_pos = 0

  for i in xrange(len(roidb)):
    if i%10==0:
      print('processing', i, file=sys.stderr)
    roi = roidb[i]
    boxes = get_boxes(roi, args.pyramid)
    gt_boxes = roidb[i]['boxes'].copy()
    gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
    num_pos += gt_boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(np.float), gt_boxes.astype(np.float))
    #print(im_info, gt_boxes.shape, boxes.shape, overlaps.shape, file=sys.stderr)

    _gt_overlaps = np.zeros((gt_boxes.shape[0]))

    if boxes.shape[0]>0:
      _gt_overlaps = overlaps.max(axis=0)
      #print('max_overlaps', _gt_overlaps, file=sys.stderr)
      for j in range(len(_gt_overlaps)):
        if _gt_overlaps[j]>config.TEST.IOU_THRESH:
          continue
        print(j, 'failed', gt_boxes[j],  'max_overlap:', _gt_overlaps[j], file=sys.stderr)

      # append recorded IoU coverage level
      found = (_gt_overlaps > config.TEST.IOU_THRESH).sum()
      _recall = found / float(gt_boxes.shape[0])
      print('recall', _recall, gt_boxes.shape[0], boxes.shape[0], gt_areas, file=sys.stderr)
      overall[0]+=found
      overall[1]+=gt_boxes.shape[0]
      #gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))
      #_recall = (gt_overlaps >= threshold).sum() / float(num_pos)
      _recall = float(overall[0])/overall[1]
      print('recall_all', _recall, file=sys.stderr)

    _vec = roidb[i]['image'].split('/')
    out_dir = os.path.join(output_folder, _vec[-2])
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_file = os.path.join(out_dir, _vec[-1].replace('jpg', 'txt'))
    with open(out_file, 'w') as f:
      name = '/'.join(roidb[i]['image'].split('/')[-2:])
      f.write("%s\n"%(name))
      f.write("%d\n"%(boxes.shape[0]))
      for b in range(boxes.shape[0]):
        box = boxes[b]
        f.write("%d %d %d %d %g \n"%(box[0], box[1], box[2]-box[0], box[3]-box[1], box[4]))

  print('Evaluating detections using official WIDER toolbox...')
  path = os.path.join(os.path.dirname(__file__), 'wider_eval_tools')
  eval_output_path = os.path.join(path, 'wider_plots')
  if not os.path.isdir(eval_output_path):
      os.mkdir(eval_output_path)
  cmd = 'cd {} && '.format(path)
  cmd += 'matlab -nodisplay -nodesktop '
  cmd += '-r "dbstop if error; '
  cmd += 'wider_eval(\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
      .format(args.output, args.method_name, eval_output_path)
  print('Running:\n{}'.format(cmd))
  subprocess.call(cmd, shell=True)

def main():
    global args
    args = parse_args()
    logger.info('Called with argument: %s' % args)
    test(args)

if __name__ == '__main__':
    main()

