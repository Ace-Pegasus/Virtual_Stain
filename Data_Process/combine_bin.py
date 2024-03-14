import os
import cv2
import numpy as np
import sys
sys.path.append('.')

def x20_to_x40():
    cluster_size, nucleus_size, overlap = 2400, 512, 32
    cluster_path = '../mask_rcnn/results/cluster' # binary map for membrane
    nucleus_path = '../mask_rcnn/results/nucleus' # binary map for nucleus
    clusterx40_path = '../mask_rcnn/results/clusterx40' # dir to save x40 membrane binary map
    img_path = '../mask_rcnn/results/cluster.jpg' # path to save the stitched membrane binary map
    save_path = '../mask_rcnn/results/bin' # dir to save mask
    tif_name = 'tumor_110' # tif name
    cluster_step = cluster_size - overlap
    nucleus_step = nucleus_size - overlap
    os.makedirs(clusterx40_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    w_start, w_end, h_start, h_end = 15701, 47103, 7168, 60928

    col_first = True
    for i in range(h_start, h_end, cluster_step):
        row_first = True
        for j in range(w_start, w_end, cluster_step):
            cluster = cv2.imread(os.path.join(cluster_path, '%s_%d-%d.jpg'%(tif_name, j // cluster_step, i // cluster_step)))
            if row_first:
                row_last = cluster
                row_first = False
            else:
                overlap1 = row_last[:, -overlap//2:]
                overlap2 = cluster[:, :overlap//2]
                merged_overlap = np.maximum(overlap1, overlap2)
                row_last = np.hstack((row_last[:, :-overlap//2], merged_overlap, cluster[:, overlap//2:]))
        if col_first:
            img_last = row_last
            col_first = False
        else:
            overlap1 = img_last[-overlap//2:, :]
            overlap2 = row_last[:overlap//2, :]
            merged_overlap = np.maximum(overlap1, overlap2)
            img_last = np.vstack((img_last[:-overlap//2, :], merged_overlap, row_last[overlap//2:, :]))
    cv2.imwrite(img_path, img_last)

    h, w = img_last.shape[:2]

    img_last = cv2.resize(img_last, (2 * w, 2 * h), interpolation=cv2.INTER_NEAREST)

    for i in range(h_start, h_end, nucleus_step):
        for j in range(w_start, w_end, nucleus_step):
            clusterx40 = img_last[i - h_start : i - h_start + nucleus_size, j - w_start : j - w_start + nucleus_size]
            cv2.imwrite(os.path.join(clusterx40_path, '%s_%d-%d.jpg'%(tif_name, j // nucleus_step, i // nucleus_step)), clusterx40)
            nucleus = cv2.imread(os.path.join(nucleus_path, '%s_%d-%d.jpg'%(tif_name, j // nucleus_step, i // nucleus_step)))
            imgray = cv2.cvtColor(nucleus, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, 127, 255, 0)
            contours_inside, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            imgray = cv2.cvtColor(clusterx40, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, 127, 255, 0)
            contours_outside, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            draw_list = []
            for contour in contours_inside:
                inside = False
                for out_contour in contours_outside:
                    not_break = True
                    for point in contour:
                        if(len(point) != 1):
                            print(len(point))
                        point = tuple(point[0])
                        point = (float(point[0]), float(point[1]))
                        if (cv2.pointPolygonTest(out_contour, point, False) != 1):
                            not_break = False
                            break
                    if not_break:
                        inside = True
                        break
                if inside:
                    draw_list.append(contour)
            combined = cv2.drawContours(clusterx40, draw_list, -1, (0, 0, 0), -1)
            combined = np.where(combined > 100, 255, combined)
            combined = np.where(combined < 100, 0, combined)
            cv2.imwrite(os.path.join(save_path, '%s_%d-%d.jpg'%(tif_name, j // nucleus_step, i // nucleus_step)), combined)

if __name__ == '__main__':
    x20_to_x40()
