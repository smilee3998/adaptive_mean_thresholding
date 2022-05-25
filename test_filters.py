import os

import cv2

from adaptive_threshold_filter import AdaptiveMeanThresholdFilter as AMTfilter
from homomorphic_filter import restore_by_homomorphic_filter

basic_threshold_value = 128
block_size = 5
sd_threshold = 20
step_size = 2
increase_block = True
offset=0
max_block_size = 13
output_path = 'output_images/'

def main():
    img1 = cv2.imread('test_images/20220514_153418_resize_warp.jpg', cv2.IMREAD_GRAYSCALE)
    amt_filter = AMTfilter(block_size=block_size, 
                    increase_block=increase_block, 
                    sd_threshold=sd_threshold, 
                    step_size=step_size, 
                    offset=offset,
                    max_block_size=max_block_size)
    
    amt_output = amt_filter.apply_filter(img1)
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    cv2.imwrite(output_path + 'test_img1.jpg', amt_output)
    
    restore_by_homomorphic_filter(img1, 'img1')      
    
if __name__ == '__main__':
    main()
    
