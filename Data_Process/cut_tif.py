import cv2
import numpy as np
import os
import openslide

def cut_tif(multi=1):
    overlap = 32
    patch_size = 2400
    overlap *= multi
    patch_size *= multi
    patch_step = patch_size - overlap
    level = 0

    tif_path = '../camelyon/dataset/train/tumor/tumor_110.tif'
    tif_name = 'tumor_110'
    save_path = '../mask_rcnn/camelyon/%d_patch'%(patch_size)
    os.makedirs(save_path, exist_ok=True)
    slide = openslide.open_slide(tif_path)

    image_size = slide.level_dimensions[level]

    print("tif_size:", image_size)
    
    w_start, w_end, h_start, h_end = 10000, 20000, 10000, 20000 # coordinates for tif

    print(f'cut_region: [[{w_start}, {w_end}], [{h_start}, {h_end}]]')

    region = np.array(slide.read_region((w_start, h_start), level, (w_end - w_start, h_end - h_start)))
    region = cv2.cvtColor(region, cv2.COLOR_RGB2BGR)
    region = cv2.resize(region, (region.shape[1] // 10, region.shape[0] // 10))
    cv2.imwrite('../mask_rcnn/camelyon/%d.jpg'%(patch_size), region)

    for i in range(w_start, w_end, patch_step):
        for j in range(h_start, h_end, patch_step):
            region = np.array(slide.read_region((i - overlap // 2, j - overlap // 2), level, (patch_size, patch_size)))
            region = cv2.cvtColor(region, cv2.COLOR_RGB2BGR)
            region = cv2.resize(region, (region.shape[1] // 2, region.shape[0] // 2))
            cv2.imwrite('%s/%s_%d-%d.jpg'%(save_path, tif_name, i // patch_step, j // patch_step), region)

if __name__ == '__main__':
    cut_tif()