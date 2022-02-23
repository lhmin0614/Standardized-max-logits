"""
Save Images
"""
from __future__ import absolute_import
from __future__ import division
import argparse
import logging
import os
import torch


import numpy as np
import random
import tensorflow as tf
import tensorflow_datasets as tfds
import bdlb
from PIL import Image
from tqdm import tqdm



def main():

    """
    Main Function
    """

    fs = bdlb.load(benchmark="fishyscapes", download_and_prepare=False)
    ds = fs.get_dataset('LostAndFound')
    
    for i, blob in tqdm(enumerate(ds)):
        image = blob['image_left'].numpy()
        img = Image.fromarray(image, 'RGB')
        img.save('/data1/leehyemin/Fishyscapes/leftImg8bit_trainvaltest/leftImg8bit/val/'+str(i)+'.png')
        
        ood_gts = blob['mask'].numpy()[..., 0]
        ood = Image.fromarray(ood_gts, 'L')
        ood.save('/data1/leehyemin/Fishyscapes/gtFine_trainvaltest/gtFine/val/'+str(i)+'.png')


if __name__ == '__main__':
    main()

