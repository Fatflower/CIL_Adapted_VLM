from torchvision import transforms
from datasets.idata import iData
import os
import numpy as np

import json
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]

class CUB200_I2T_OOD(iData):
    '''
    Dataset Name:   CUB200-2011
    Task:           fine-grain birds classification
    Data Format:    224x224 color images. (origin imgs have different w,h)
    Data Amount:    5,994 images for training and 5,794 for validationg/testing
    Class Num:      200
    Label:          

    Reference:      https://opendatalab.com/CUB-200-2011
    '''
    def __init__(self, img_size=None) -> None:
        super().__init__()
        self.use_path = True
        self.img_size = 224 if img_size is None else img_size
        self.train_trsf = [
            transforms.RandomResizedCrop(224, (0.6, 1)),
            transforms.RandomRotation((0, 10)),
            transforms.RandomHorizontalFlip(),
            ]
        self.strong_trsf = [
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
        ]
        self.test_trsf = []


        self.common_trsf = [
            transforms.Resize((self.img_size, self.img_size), interpolation=BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]

        self.class_order = np.arange(200).tolist()
        self.class_to_idx = {}

    def getdata(self, train:bool, root_dir, img_dir):
        data, targets = [], []
        with open(os.path.join(root_dir, 'train_test_split.txt')) as f:
            for line in f:
                image_id, is_train = line.split()
                if int(is_train) == int(train):
                    data.append(os.path.join(img_dir, self.images_path[image_id]))
                    targets.append(self.class_ids[image_id])

        return np.array(data), np.array(targets)

    def download_data(self):
        root_dir = os.path.join(os.environ["DATA"], 'CUB_200_2011')
        img_dir = os.path.join(root_dir, 'images_OOD_28')

        self.images_path = {}
        with open(os.path.join(root_dir, 'images.txt')) as f:
            for line in f:
                image_id, path = line.split()
                self.images_path[image_id] = path

        self.class_ids = {}
        with open(os.path.join(root_dir, 'image_class_labels.txt')) as f:
            for line in f:
                image_id, class_id = line.split()
                self.class_ids[image_id] = int(class_id) - 1

        with open(os.path.join(root_dir, 'classes.txt')) as f:
            for line in f:
                class_id, class_name = line.split()
                # self.class_to_idx[class_name] = int(class_id) - 1
                self.class_to_idx[class_name[4:]] = int(class_id) - 1

        self.train_data, self.train_targets = self.getdata(True, root_dir, img_dir)
        self.test_data, self.test_targets = self.getdata(False, root_dir, img_dir)

        # print(len(np.unique(self.train_targets))) # output: 200
        # print(len(np.unique(self.test_targets))) # output: 200