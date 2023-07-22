# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 10:02:26 2022

@author: h.GONG
"""

from PIL import Image
import glob, os
from tqdm import tqdm


def Img_Resize(images_name, out_path):
    
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    
    for image in tqdm(images_name):
        img = Image.open(image)
        out = img.resize((128, 128), Image.ANTIALIAS)
        out.save(os.path.join(out_path, os.path.basename(image)))


if __name__ == "__main__":
    
    images_name = glob.glob(r'D:\tools\anaconda\Proj\GAN\dataset512\val\*.jpg')
  
    out_path = r'D:\tools\anaconda\Proj\GAN\dataset128\val'
    
    Img_Resize(images_name, out_path)

    print('\nDone! Output path is: ' + out_path)