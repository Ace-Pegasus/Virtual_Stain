import os
import cv2
import argparse
import matplotlib.pyplot as plt
from mask_rcnn import segmentation_model, plot_masks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, help='path to your test image')
    parser.add_argument('--model', type=str, help='path to saved model')

    args = parser.parse_args()
    
    IMAGE_PATH = args.img_path
    MODEL_PATH = args.model

    save_path = './results'
    os.makedirs(save_path, exist_ok=True)

    classes = {0:"background", 1:"nucleus"}
    num_classes = 2
    model = segmentation_model(MODEL_PATH,num_classes)
    
    image_list = os.listdir(IMAGE_PATH)

    for image_name in image_list:
        image_path = os.path.join(IMAGE_PATH, image_name)

        image = cv2.imread(image_path)
        pred = model.detect_masks(image, rgb_image=False)   # rgb_image=False if loading image with cv2.imread()

        plotted = plot_masks(image,pred,classes)
        
        cv2.imwrite(f'./{save_path}/{image_name}', plotted)
