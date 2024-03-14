import os
import numpy as np
import cv2
import torch
import torch.utils.data
import utils.transforms as T
from PIL import Image

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class maskrcnn_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        # self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
        # self.masks = list(sorted(os.listdir(os.path.join(root, "SegmentationObject"))))
        # self.class_masks = list(sorted(os.listdir(os.path.join(root, "SegmentationClass"))))
        self.img_path = []
        self.masks_path = []
        for sample in os.listdir(root):
            sample_path = os.path.join(root, sample)
            self.img_path.append(os.path.join(sample_path, 'image', os.listdir(os.path.join(sample_path, 'image'))[0]))
            mask_path = [os.path.join(sample_path, 'mask', mask) for mask in os.listdir(os.path.join(sample_path, 'mask'))]
            self.masks_path.append(mask_path)

    def __getitem__(self, idx):
        # load images ad masks
        # img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        # mask_path = os.path.join(self.root, "SegmentationObject", self.masks[idx])
        # class_mask_path = os.path.join(self.root, "SegmentationClass", self.class_masks[idx])

        img_path = self.img_path[idx]
        masks_path = self.masks_path[idx]
        
        #read and convert image to RGB
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        # mask = Image.open(mask_path)
        
        # mask = cv2.imread(mask_path,0)
        # class_mask = Image.open(class_mask_path).convert('P')
        # class_mask = np.asarray(class_mask)
        # # instances are encoded as different colors
        # obj_ids = np.unique(mask)
        # # first id is the background, so remove it
        # obj_ids = obj_ids[1:]

        # # split the color-encoded mask into a set
        # # of binary masks
        # masks = mask == obj_ids[:, None, None]

        masks = []

        for mask_path in masks_path:
            mask = cv2.imread(mask_path, 0)
            _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
            masks.append(mask)

        # get bounding box coordinates for each mask
        num_objs = len(masks)
        boxes = []
        invalid_mask = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if xmax - xmin <= 0 or ymax - ymin <= 0:
                invalid_mask.append(i)
            boxes.append([xmin, ymin, xmax, ymax])

        # there is only one class
        labels = np.array([])
        for i in range(len(masks)):
            labels = np.append(labels, 1)

        masks = [e for i, e in enumerate(masks) if i not in invalid_mask]
        boxes = [e for i, e in enumerate(boxes) if i not in invalid_mask]
        labels = [e for i, e in enumerate(labels) if i not in invalid_mask]
        
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # print(boxes.shape)
        # print(labels.shape)
        # print(masks.shape)
        # print(img.shape)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_path)