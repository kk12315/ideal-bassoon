from glob import glob

import cv2

from hand_tracker import HandTracker

palm_model_path = "./models/palm_detection.tflite"
landmark_model_path = "./models/hand_landmark.tflite"
anchors_path = "./data/anchors.csv"

# box_shift determines
detector = HandTracker(palm_model_path, landmark_model_path, anchors_path,
                       box_shift=0.2, box_enlarge=1.3)

def draw_boxes(image, box):
    image = cv2.rectangle(image, tuple(box[0]), tuple(box[2]), (255,255,255))
    return image

def find_x_y(box):
    box[box<0]=0
    x1 = min(box[:,0])
    x2 = max(box[:, 0])
    y1 = min(box[:, 1])
    y2 = max(box[:, 1])
    return x1, x2,y1,y2

root_path = 'cropped'
import os
from tqdm import tqdm


hand_imgs = glob('dataset/*/*/*.jpeg')

for path in tqdm(hand_imgs):
    img = cv2.imread(path)[:,:,::-1]
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (256, 256))
    kp, box = detector(img)
    if kp is None:
        continue
    coor = find_x_y(box)
    x1, x2, y1, y2 = map(int, coor)
    img = img[:,:,::-1]
    img = img[y1:y2, x1:x2, :]
    path = os.path.join(root_path, path)
    img = cv2.flip(img, 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)
    # cv2.imshow("vsdf", img)
    # cv2.waitKey(0)
