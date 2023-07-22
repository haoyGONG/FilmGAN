# -*- coding: utf-8 -*-
"""
Created on Tue May 31 10:44:21 2022

@author: 10970
"""

from PIL import Image
import glob
import numpy as np


def combine(filepath = './all_val/*.png'):

  filepath = filepath
  imgs = glob.glob(filepath)
  imgs = np.array(imgs).reshape(-1, 3)
  file_num = 0
  
  for img in imgs:
      
      fake = Image.open(img[2])
      ref = Image.open(img[1]).resize((768, 512), Image.BILINEAR)
      src = Image.open(img[0])
  
      result = Image.new(src.mode, (src.size[0]*2, src.size[0]*2), (255,255,255))
      result.paste(src, box= (src.size[0],0))
      result.paste(ref, box= (0,ref.size[0]))
      result.paste(fake, box = (fake.size[0], fake.size[0]))
      save_name = img[0].split('_')[-1].split('.')[0] + '_' + img[1].split('_')[-1].split('.')[0]
      result.save(f'./result/{file_num}_{save_name}.png')
      
      file_num += 1










