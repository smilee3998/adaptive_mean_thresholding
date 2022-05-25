import os
from pathlib import Path
from typing import List

import cv2
import numpy as np


class HomomorphicFilter:
    def __init__(self, gammaL=0.5, gammaH=2, cutoff=10):
        """
        Args:
            gammaL: the gain of the low frequency compoenent
            gammaH: the gain of he high frequency component
            cutoff: frequency threshold for the high pass filter 
        """
        self.gammaL = gammaL
        self.gammaH = gammaH
        self.D0 = cutoff

    def filter_img(self, img: np.ndarray):
        img_shape = img.shape
        if len(img_shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img_shape) != 2:
            raise ValueError(f"invalid image shape {len(img_shape)=}")

        filter = self.gaussian_high_emphasis_filter(img_shape)

        log_img = np.log(1 + img.astype(np.float64))
        fft_img = np.fft.fft2(log_img)

        result = filter * fft_img
        result = np.fft.ifft2(result)
        result = np.exp(np.real(result)) - 1
        result[result > 255] = 255
        result = result.astype(np.uint8)

        return result

    def gaussian_high_emphasis_filter(self, img_shape):
        return np.fft.fftshift(self.gammaL + self.gammaH * self.gaussian_high_pass_filter(img_shape))

    def gaussian_high_pass_filter(self, img_shape):
        if self.D0 == 0:
            return np.ones(img_shape)

        center_x = img_shape[0] // 2
        center_y = img_shape[1] // 2

        filter = np.empty_like(img_shape)
        X, Y = np.meshgrid(np.arange(img_shape[0]),
                           np.arange(img_shape[1]),
                           sparse=False,
                           indexing='ij')
        distance = ((X-center_x)**2 + (Y - center_y)**2).astype(np.float64)
        filter = 1 - np.exp(-distance / (2*self.D0**2))
        return filter


def restore_by_homomorphic_filter(img, output_filename: str,
                                  gammaLs: List = [0.6, 0.7],
                                  gammaHs: List = [1.5, 2.0],
                                  cutoffs: List = [10, 30],
                                  write_img=True,
                                  output_path='output_images/',
                                  ):
    for gammaL in gammaLs:
        for gammaH in gammaHs:
            for cutoff in cutoffs:
                homo_filter = HomomorphicFilter(
                    gammaL=gammaL, gammaH=gammaH, cutoff=cutoff)
                ghef_filtered_img = homo_filter.filter_img(img)
                if not os.path.isdir(output_path) and write_img:
                    os.mkdir(output_path)

                if write_img:
                    cv2.imwrite(
                        output_path + f'{output_filename}_{gammaL=}_{gammaH=}_{cutoff=}.jpg', ghef_filtered_img)
