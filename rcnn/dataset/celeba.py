# --------------------------------------------------------
# SSH: Single Stage Headless Face Detector
# CelebA Dataset Loader
# Written by Shawn Liu
# --------------------------------------------------------
from __future__ import print_function
try:
    import cPickle as pickle
except ImportError:
    import pickle
import cv2
import os
import numpy as np
from PIL import Image

from config import config
from logger import logger
from .imdb import IMDB
from .pascal_voc_eval import voc_eval
from .ds_utils import unique_boxes, filter_small_boxes

class celeba(IMDB):
    def __init__(self, image_set, root_path, data_path):
        super(celeba, self).__init__('CELEBA', image_set, root_path, data_path)

        split = image_set
        self._split = image_set
        self._image_set = image_set


        self._anno_bbox_file_name = 'list_bbox_celeba.txt'
        self._anno_landmark_file_name = 'list_landmarks_celeba.txt'
        self.root_path = root_path
        self.data_path = data_path


        self._dataset_path = self.data_path
        self._imgs_path = os.path.join(self._dataset_path, 'Img', 'img_celeba')

        # Read the annotations file
        anno_bbox_path = os.path.join(self._dataset_path,'Anno',self._anno_bbox_file_name)
        anno_landmark_path = os.path.join(self._dataset_path,'Anno',self._anno_landmark_file_name)
        assert os.path.isfile(anno_bbox_path), 'Annotation file not found {}'.format(anno_bbox_path)
        assert os.path.isfile(anno_landmark_path), 'Annotation file not found {}'.format(anno_landmark_path)

        with open(anno_bbox_path, 'r') as file:
            annos_bbox = file.readlines()
        with open(anno_landmark_path, 'r') as file:
            annos_landmark = file.readlines()

        self._fp_bbox_map = {}
        count = 2
        while count < len(annos_bbox):
            name = annos_bbox[count].split()[0]
            self._fp_bbox_map[name] = []
            bbox = annos_bbox[count].split()[1:5]
            bbox = [int(round(float(x))) for x in bbox]
            x1 = max(0, bbox[0])
            y1 = max(0, bbox[1])
            bbox = [x1, y1, x1 + bbox[2], y1 + bbox[3]]
            landmark = annos_landmark[count].split()[1:11]
            landmark = [int(round(float(x))) for x in landmark]
            self._fp_bbox_map[name].append([bbox, landmark])
            count += 1
        self._image_paths = self._fp_bbox_map.keys()


        #self.num_images = len(self._image_paths)
        self._image_index = range(len(self._image_paths))
        self.classes = ['bg', 'face']
        self.num_classes = len(self.classes)


    def _get_default_path(self):
        return os.path.join(cfg.DATA_DIR,'datasets', 'wider')

    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        image_path = os.path.join(self._imgs_path, self._image_paths[index])
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def gt_roidb(self):
        cache_file = os.path.join(self.cache_path, '{}_{}_gt_roidb.pkl'.format(self.name, self._split))
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            self.num_images = len(roidb)
            return roidb

        roidb = []
        max_num_boxes = 0

        for fp in self._image_paths:
            boxes = np.zeros([len(self._fp_bbox_map[fp]), 4], np.float)
            boxes_mask = []
            landmarks = np.zeros([len(self._fp_bbox_map[fp]), 10], np.float)

            gt_classes = np.ones([len(self._fp_bbox_map[fp])], np.int32)
            overlaps = np.zeros([len(self._fp_bbox_map[fp]), 2], np.float)

            ix = 0

            for bbox, landmark in self._fp_bbox_map[fp]:
                imsize = Image.open(os.path.join(self._imgs_path, fp)).size

                x1 = bbox[0]
                y1 = bbox[1]
                x2 = min(imsize[0], bbox[2])
                y2 = min(imsize[1], bbox[3])

                if self._image_set=='train':
                    if (x2 - x1) < config.TRAIN.MIN_BOX_SIZE or y2 - y1 < config.TRAIN.MIN_BOX_SIZE:
                        continue
                    if config.BBOX_MASK_THRESH>0:
                        if (x2 - x1) < config.BBOX_MASK_THRESH or y2 - y1 < config.BBOX_MASK_THRESH:
                            boxes_mask.append(np.array([x1, y1, x2, y2], np.float))
                            continue
                else:
                    if (x2 - x1) < 1 or y2 - y1 < 1:
                        continue

                boxes[ix, :] = np.array([x1, y1, x2, y2], np.float)
                landmarks[ix, :] = landmark

                # img = cv2.imread(os.path.join(self._imgs_path, fp))
                # cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # for i in range(5):
                #     cv2.circle(img, (int(landmark[2*i]), int(landmark[2*i+1])), 1, (0, 0, 255), 2)
                # cv2.imshow("img", img)
                # cv2.waitKey(0)

                cls = int(1)
                gt_classes[ix] = cls
                overlaps[ix, cls] = 1.0
                ix += 1
            max_num_boxes = max(max_num_boxes, ix)
            #overlaps = scipy.sparse.csr_matrix(overlaps)
            if self._image_set=='train' and ix==0:
                continue
            boxes = boxes[:ix,:]
            landmarks = landmarks[:ix,:]
            gt_classes = gt_classes[:ix]
            overlaps = overlaps[:ix,:]
            image_path = os.path.join(self._imgs_path, fp)
            with open(image_path, 'rb') as fin:
                stream = fin.read()
            stream = np.fromstring(stream, dtype=np.uint8)

            roi = {
                'image': image_path,
                'stream': stream,
                'height': imsize[1],
                'width': imsize[0],
                'boxes': boxes,
                'landmarks': landmarks,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'max_classes': overlaps.argmax(axis=1),
                'max_overlaps': overlaps.max(axis=1),
                'flipped': False,
            }
            if len(boxes_mask)>0:
                boxes_mask = np.array(boxes_mask)
                roi['boxes_mask'] = boxes_mask
            roidb.append(roi)
        for roi in roidb:
          roi['max_num_boxes'] = max_num_boxes
        self.num_images = len(roidb)
        with open(cache_file, 'wb') as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return roidb

    def write_detections(self, all_boxes, output_dir='./output/'):

        print('Writing the detections to text files: {}...'.format( output_dir), end='')
        for i in range(len(self._image_paths)):
            img_path = self._image_paths[i]

            img_name = os.path.basename(img_path)
            img_dir = img_path[:img_path.find(img_name) - 1]

            txt_fname = os.path.join(output_dir, img_dir, img_name.replace('jpg', 'txt'))

            res_dir = os.path.join(output_dir, img_dir)
            if not os.path.isdir(res_dir):
                os.makedirs(res_dir)

            with open(txt_fname, 'w') as f:
                f.write(img_path + '\n')
                f.write(str(len(all_boxes[1][i])) + '\n')
                for det in all_boxes[1][i]:
                    f.write('%d %d %d %d %g \n' % (
                        int(det[0]), int(det[1]), int(det[2]) - int(det[0]), int(det[3]) - int(det[1]),
                        det[4]))
        print('Done!')

        
    def evaluate_detections(self, all_boxes, output_dir='./output/',method_name='insightdetection'):
        detections_txt_path = os.path.join(output_dir,'detections')
        self.write_detections(all_boxes,detections_txt_path)

        print('Evaluating detections using official WIDER toolbox...')
        path = os.path.join(os.path.dirname(__file__),
                                '..',  'wider_eval_tools')
        eval_output_path = os.path.join(output_dir,'wider_plots')
        if not os.path.isdir(eval_output_path):
            os.mkdir(eval_output_path)
        cmd = 'cd {} && '.format(path)
        cmd += 'matlab -nodisplay -nodesktop '
        cmd += '-r "dbstop if error; '
        cmd += 'wider_eval(\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(detections_txt_path, method_name,eval_output_path)
        print('Running:\n{}'.format(cmd))
        subprocess.call(cmd, shell=True)

