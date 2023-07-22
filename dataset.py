from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import config
import random

def listdir(dname):
    
    fnames = os.listdir(dname)
    for i in range(len(fnames)):
        fnames[i] = os.path.join(dname, fnames[i])
    
    return fnames

class RefDataset(Dataset):
    def __init__(self, root, transform=None):
        
        domains = os.listdir(root)
        self.fnames, self.fnames2, self.labels = [], [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            self.fnames += cls_fnames
            self.fnames2 += random.sample(cls_fnames, len(cls_fnames))
            self.labels += [idx] * len(cls_fnames)
        
        self.transform = transform
        self.length_dataset = len(self.fnames)
        

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        
        label = self.labels[index]
        img = np.array(Image.open(self.fnames[index]).convert("RGB"))
        img2 = np.array(Image.open(self.fnames2[index]).convert("RGB"))


        if self.transform:
            augmentations = self.transform(image=img, image0=img2)
            img = augmentations["image"]
            img2 = augmentations["image0"]

        return img, img2, label
    
    
    
class MapDataset(Dataset):
    def __init__(self, root, transform=None):
        
        domains = os.listdir(root)
        self.fnames, self.labels = [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            self.fnames += cls_fnames
            self.labels += [idx] * len(cls_fnames)
        
        self.transform = transform
        self.length_dataset = len(self.fnames)
        

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        
        label = self.labels[index]
        img = np.array(Image.open(self.fnames[index]).convert("RGB"))
        img2 = np.array(Image.open(self.fnames[index]).convert("RGB"))


        if self.transform:
            augmentations = self.transform(image=img, image0=img2)
            img = augmentations["image"]
            img2 = augmentations["image0"]

        return img, label

class ValRefDataset(Dataset):
    def __init__(self, root, transform=None):
        
        domains = os.listdir(root)
        self.fnames, self.labels = [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            self.fnames.append(cls_fnames)
            self.labels.append([idx] * len(cls_fnames))
        
        self.transform = transform
        self.length_dataset = len(self.fnames[0])
        

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        
        label = self.labels[0][index]
        label2 = self.labels[1][index]
        label3 = self.labels[2][index]
        img = np.array(Image.open(self.fnames[0][index]).convert("RGB"))
        img2 = np.array(Image.open(self.fnames[1][index]).convert("RGB"))
        img3 = np.array(Image.open(self.fnames[2][index]).convert("RGB"))
        img4 = np.array(Image.open(self.fnames[2][index]).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=img, image0=img2)
            img = augmentations["image"]
            img2 = augmentations["image0"]
            augmentations2 = self.transform(image=img3, image0=img4)
            img3 = augmentations2["image"]
            img4 = augmentations2["image0"]

        return img, img2, img3, label, label2, label3


def main():
    from torch.utils.data import DataLoader
    import encoder
    root_src = r'D:\tools\anaconda\Proj\GAN\dataset\train\src\portra160'
    root_ref = r'D:\tools\anaconda\Proj\GAN\dataset\train\src'
    ref = MapDataset(root_ref, config.transforms)
   
    
    loader_ref = DataLoader(
        ref,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    
   
            
    for idx, (x, y) in enumerate(loader_ref):
        
        if idx == 0:
            print(type(x))
            print(x)
            print(y)
        if idx == 1200:
            print(x)
            print(y)



if __name__ == '__main__':
    main()

    