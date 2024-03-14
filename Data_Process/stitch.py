import cv2
import os
import numpy as np

model_index = 'latest'
model = '' # CycleGAN model

save_path = '../CycleGAN/stitch_result/images'
if not os.path.exists(save_path):
    os.mkdir(save_path)

w_start_, h_start_ = 11000, 84000
w_end_, h_end_ = 23600, 99000

img_path = '../CycleGAN/results/%s/test_%s/images'%(model, model_index)

img_name = '' # WSI name

overlap = 128
patch_size = 512
patch_step = patch_size - overlap
col_start = w_start_ // patch_step
col_end = w_end_ // patch_step
row_start = h_start_ // patch_step
row_end = h_end_ // patch_step

weights = np.linspace(0, 1, overlap, endpoint=True)
weights_row = np.expand_dims(weights, axis=-1).repeat(3, axis=-1)[np.newaxis, :, :]
weights_col = weights_row.transpose(1, 0, 2)

if not os.path.exists(save_path):
    os.mkdir(save_path)

def row_stitch(img1, img2):
    img1_left = img1[:, : -overlap]
    img1_right = img1[:, -overlap :]
    img2_left = img2[:, : overlap]
    img2_right = img2[:, overlap :]
    img_fusion = img1_right * (1 - weights_row) + img2_left * weights_row
    img = np.hstack((img1_left, img_fusion, img2_right))
    return img

def col_stitch(img1, img2):
    img1_up = img1[: -overlap, :]
    img1_down = img1[-overlap : , :]
    img2_up = img2[: overlap, :]
    img2_down = img2[overlap : , :]
    img_fusion = img1_down * (1 - weights_col) + img2_up * weights_col
    img = np.vstack((img1_up, img_fusion, img2_down))
    return img


def stitch():
    img_last = []
    col_first = True
    for i in range(row_start, row_end):
        row_last = []
        row_first = True
        for j in range(col_start, col_end):
            bin_img = cv2.imread(os.path.join(img_path, '%s_%d-%d_fake_B.png'%(img_name, j, i)))
            if row_first:
                row_last = bin_img
                row_first = False
            else:
                row_last = row_stitch(row_last, bin_img)
        if col_first:
            img_last = row_last
            col_first = False
        else:
            img_last = col_stitch(img_last, row_last)
    cv2.imwrite('%s/%s-ck-%s.jpg'%(save_path, img_name, model), img_last)


if __name__ == '__main__':
    stitch()
