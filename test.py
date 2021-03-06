import cv2
import sys
import numpy as np
import datetime
#sys.path.append('.')
from essh_detector import ESSHDetector

scales = [1200, 1600]
#scales = [600, 1200]
t = 10
detector = ESSHDetector('./model/essh-r50', 0, ctx_id=0)

def main():
  f = './sample-images/t1.jpg'
  if len(sys.argv)>1:
    f = sys.argv[1]
  img = cv2.imread(f)
  im_shape = img.shape
  print(im_shape)
  target_size = scales[0]
  max_size = scales[1]
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])
  if im_size_min>target_size or im_size_max>max_size:
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
    print('resize to', img.shape)

  for i in xrange(t-1): #warmup
    faces = detector.detect(img)
  timea = datetime.datetime.now()

  faces = detector.detect(img, threshold=0.5)
  bbox = np.round(faces[:,0:5])
  landmark = faces[:, 5:15].reshape(-1,5,2)

  timeb = datetime.datetime.now()
  diff = timeb - timea
  print('detection uses', diff.total_seconds(), 'seconds')
  print('find', faces.shape[0], 'faces')
  # print(bbox)
  # print(landmark)
  # for i in xrange(faces.shape[0]):
  #   cv2.rectangle(img, (faces[i,0],faces[i,1]), (faces[i,2],faces[i,3]), (0,255,0), 2)
  # cv2.imshow('faces', img)
  # cv2.waitKey(0)

  for b in bbox:
    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
  for p in landmark:
    for i in range(5):
      cv2.circle(img, (p[i][0], p[i][1]), 1, (0, 0, 255), 2)
  cv2.imshow("detection result", img)
  cv2.waitKey(0)

if __name__ == '__main__':
    main()