# load dataset and  calculate mean and std
import os

import numpy as np
import torch
from PIL import Image

from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms


class MultiLabelRGBDataSet(torch.utils.data.Dataset):
    def __init__(self, imgspath, imgslist, annotationpath, transforms=None):
        self.imgslist = imgslist
        self.imgspath = imgspath
        self.transform = transforms
        self.annotationpath = annotationpath
        # print(annotationpath)

    def __len__(self):
        return len(self.imgslist)

    def __getitem__(self, index):
        ipath = os.path.join(self.imgspath, self.imgslist[index])
        img = Image.open(ipath)
        # print(ipath)
        if self.transform is not None:
            img = self.transform(img)
        (filename, extension) = os.path.splitext(ipath)
        filename = os.path.basename(filename)
        annotation = os.path.join(self.annotationpath, filename + ".txt")
        label = np.loadtxt(annotation, dtype=np.int64)
        return img, label, filename

trans = transforms.Compose(([
    transforms.Resize((224,224)),
    transforms.ToTensor()  # divides by 255
]))

rgb_dir = '/home/kai/Desktop/RGB'
label_dir = '/home/kai/Desktop/yoliclabel'

img_list = os.listdir(rgb_dir)
x_train, x_test = train_test_split(img_list, test_size=0.3, random_state=2)

train = MultiLabelRGBDataSet(rgb_dir,
                          x_train, label_dir, trans)

train_loader = torch.utils.data.DataLoader(train,
                                           batch_size=1,
                                           shuffle=False)

# calculate rgb mean and std in train_loader
rgb_mean = torch.zeros(3)
rgb_std = torch.zeros(3)
for i, (img, label, filename) in enumerate(train_loader):
    rgb_mean += img.mean(dim=(0, 2, 3))
    rgb_std += img.std(dim=(0, 2, 3))
    if i == len(train_loader) - 1:
        break
rgb_mean /= len(train_loader)
rgb_std /= len(train_loader)
print(rgb_mean)
print(rgb_std)




