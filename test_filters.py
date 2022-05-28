import os
import re
from pathlib import Path

import cv2
import numpy as np

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
test_orignal_path = Path('test_images/original')
test_edit_path = Path('test_images/edit')
LOGO_PIXELS = 81

def remove_logo(img, num_qr):
    if num_qr == 57:
        img[25:33+1, 25:33+1] = 255
    elif num_qr == 65:
        img[27:37+1, 27:37+1] = 255
    elif num_qr == 69:
        img[29:39+1, 29:39+1] = 255
    else:
        raise ValueError('Not support qr version')
    return img
    
def get_ground_truth(img_filename, num_qr):
    edit_img = cv2.imread(str(test_edit_path / img_filename) , cv2.IMREAD_GRAYSCALE)
    edit_img = cv2.resize(edit_img, (num_qr, num_qr))
    _, threshold = cv2.threshold(edit_img, basic_threshold_value, 255, cv2.THRESH_BINARY)
    threshold = remove_logo(threshold, num_qr)
    return threshold

def resize_img(image):
    height, width = image.shape[:]
    new_height,new_width = height, width
    
    ratio = 2
    
    while new_height + new_width > 600:
        new_height /= ratio
        new_width /= ratio
        ratio += 0.5
    print(f'{new_height=} {new_width=}')
    image = cv2.resize(image, (int(new_height), int(new_width)), cv2.INTER_AREA)
    return image

    
def main():
    amt_filter = AMTfilter(block_size=block_size, 
                    increase_block=increase_block, 
                    sd_threshold=sd_threshold, 
                    step_size=step_size, 
                    offset=offset,
                    max_block_size=max_block_size)
    
    for filepath in test_orignal_path.iterdir():
        filename = filepath.name
        # number of qr codde need to input manually
        num_qr = re.search('QR=(\d+)', filename)
        print(num_qr.group(1))
        if num_qr is None:
            continue
        num_qr = int(num_qr.group(1))
        
        img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
        img = resize_img(img)  # resize to resolution lower than 300x300
        correct = get_ground_truth(filename, num_qr)
        
        amt_output = amt_filter.apply_filter(img)
        reszie_amt_output = cv2.resize(amt_output, (num_qr, num_qr))
        _, output = cv2.threshold(reszie_amt_output, 128, 255, cv2.THRESH_BINARY)
        output = remove_logo(output, num_qr)
        
        diff = np.abs(output.astype(np.int32) - correct.astype(np.int32))
        num_diff  = np.count_nonzero(diff)
        num_pixels = num_qr * num_qr - LOGO_PIXELS  # we don't count the logo
        error_rate = num_diff / num_pixels 
        
        print(f'For image {filename}, {error_rate=} {num_diff=} ')  
        cv2.imwrite(output_path + filename, amt_output)
        
        # restore_by_homomorphic_filter(img, 'img1')      
    
if __name__ == '__main__':
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    main()
    
