import numpy as np
from scipy.stats import pearsonr
import cv2
import os
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(img1, img2):
    ssim_score = ssim(img1, img2, channel_axis=-1)
    return ssim_score

def calculate_msssim(img1, img2):
    scale_factors = [1, 2, 4, 8, 16]
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    ssim_scores = []
    for scale in scale_factors:
        resized_img1 = cv2.resize(img1, None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_LINEAR)
        resized_img2 = cv2.resize(img2, None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_LINEAR)
        ssim_score = calculate_ssim(resized_img1, resized_img2)
        ssim_scores.append(ssim_score)

    ssim_scores = np.maximum(ssim_scores, 0)
    # ssim_scores = np.where(ssim_scores == 0, 1e-10, ssim_scores)
    msssim_score = np.prod(np.power(ssim_scores, weights))

    return msssim_score

def uiqi(img1, img2):
    mu_x = np.mean(img1)
    mu_y = np.mean(img2)
    
    sigma_x = np.var(img1)
    sigma_y = np.var(img2)
    
    sigma_xy = np.cov(img1.flat, img2.flat)[0, 1]
    
    uiqi_score = (4 * sigma_xy * mu_x * mu_y) / ((sigma_x + sigma_y) * (mu_x**2 + mu_y**2))
    
    return uiqi_score

if __name__ == '__main__':
    source_dir = '' # source dataset
    target_dir = '' # generated dataset

    source_image_paths = os.listdir(source_dir)
    target_image_paths = os.listdir(target_dir)
    source_image_paths.sort()
    target_image_paths.sort()
    ms_ssim = []
    pcc = []
    uqi = []
    for i in range(len(source_image_paths)):
        source_image_path = source_image_paths[i]
        target_image_path = target_image_paths[i]
        source_image = cv2.imread(os.path.join(source_dir, source_image_path))
        target_image = cv2.imread(os.path.join(target_dir,target_image_path))
        source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        source_size = source_image.shape[1::-1]
        target_image = cv2.resize(target_image, source_size, interpolation=cv2.INTER_LINEAR)
        # print(source_image.shape)
        # print(target_image.shape)
        msssim_score = calculate_ssim(source_image, target_image)
        # msssim_score = calculate_msssim(source_image, target_image)
        ms_ssim.append(msssim_score)
        # correlation_coefficient, _ = pearsonr(source_image.flatten(), target_image.flatten())
        # pcc.append(correlation_coefficient)
        # uqi_score = uiqi(source_image, target_image)
        # uqi.append(uqi_score)

    # print(ms_ssim)
    print(sum(ms_ssim)/len(ms_ssim))
