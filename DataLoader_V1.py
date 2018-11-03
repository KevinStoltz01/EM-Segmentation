import glob
import os
import numpy as np
import pandas as pd
from PIL import Image
import random
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from torch.utils.data import DataLoader

dataset = "4373"
img_size = 512
dataset = str(dataset)
experiment = "randomCrops_nucleiTest_tooManyBlanks" + dataset + "_"
semi_superivsed = True
local = True
random_crops = True

if semi_superivsed:
    test_size1 = 0.9
    test_size2 = 0.5
if not semi_superivsed:
    test_size1 = 0.8
    test_size2 = 0.25


def train_val_split(directory, white_pix_threshold=0.95):
    names = pd.read_csv(directory + 'names.csv')
    names = names.iloc[:, 1].tolist()

    train, valtest = train_test_split(names, test_size=test_size1, random_state=random.randint(0, 100))
    val, test = train_test_split(valtest, test_size=test_size2, random_state=random.randint(0, 100))

    df_train = pd.DataFrame(train)
    df_val = pd.DataFrame(val)
    df_train.to_csv(directory + experiment + "train_names.csv")
    df_val.to_csv(directory + experiment + "val_names.csv")

    train_names = [glob.glob(directory + "Outputs\\" + name) for name in train]

    number_of_images = 0
    percent_white_pix = 0
    for i, name in enumerate(train_names):
        print("checking image [{}/{}] for white pixel content".format(i + 1, len(train_names)))
        im = Image.open(train_names[i][0])
        im_array = np.array(im)
        w = im_array.shape[0]
        h = im_array.shape[1]
        pix = h * w
        im_array = np.sum(np.where(im_array > 0, 1, 0))
        percent_white_pix += (im_array / pix) * 100
        avg_percent_white_pix = percent_white_pix / (number_of_images + 1)
        number_of_images += 1

    if avg_percent_white_pix < white_pix_threshold:
        print("Searching for images with Nucleoli")
        train_val_split(directory, white_pix_threshold)
    else:
        print("average percentage of white pixels - {}".format(avg_percent_white_pix))


class DataSet(Dataset):
    def __init__(self, names_csv, x_dir, y_dir, n_dir=None, nuc=False, transform=None):
        self.x_dir = x_dir
        self.y_dir = y_dir
        if n_dir:
            self.n_dir = n_dir
        self.names_csv = pd.read_csv(names_csv)
        self.nuc = nuc
        self.transform = transform

    def __len__(self):
        return len(self.names_csv)

    def __getitem__(self, idx):
        # Get image names
        name = self.names_csv.iloc[idx, 1]
        name = name[0:-4]

        # Load images
        x = Image.open(os.path.join(self.x_dir, self.names_csv.iloc[idx, 1]))
        y = Image.open(os.path.join(self.y_dir, self.names_csv.iloc[idx, 1]))
        if self.nuc:
            n = Image.open(os.path.join(self.n_dir, self.names_csv.iloc[idx, 1]))

        if self.nuc:
            images = {"x": x, "y": y, "n": n, "name": name}

        else:
            images = {"x": x, "y": y, "name": name}

        if self.transform:
            images = self.transform(images)

        return images


class CropAndTransform:
    def __init__(self, crop_size=512, augment=False, nuc=False, wpix_threshold=0.25, keep_prob=0.001, random_crops=True):
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        elif isinstance(crop_size, tuple):
            self.crop_size = crop_size
        else:
            raise TypeError("Variable crop_size must be type int or tuple")
        self.augment = augment
        self.nuc = nuc
        self.random_crops = random_crops
        self.wpix_threshold = wpix_threshold
        self.keep_prob = keep_prob

    def __call__(self, images):
        name = images["name"]
        x = images["x"]
        y = images["y"]
        if self.nuc:
            n = images["n"]

        #  check to see if y contains nucleoli and reset the process if it does not
        if self.random_crops:
            wpix = 0
            while wpix < self.wpix_threshold:
                # Random crops
                i, j, h, w = transforms.RandomCrop.get_params(x, output_size=self.crop_size)
                x = TF.crop(x, i, j, h, w)
                y = TF.crop(y, i, j, h, w)
                y_array = np.array(y) / 255
                if self.nuc:
                    n = TF.crop(n, i, j, h, w)
                wpix = np.sum(y_array) / (y_array.shape[0] * y_array.shape[1])
                if wpix < self.wpix_threshold:
                    keep_prob = random.random()
                    if keep_prob < self.keep_prob:
                        break
                    else:
                        x = images["x"]
                        y = images["y"]

        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                x = TF.hflip(x)
                y = TF.hflip(y)
                if self.nuc:
                    n = TF.hflip(n)

            # Random vertical flip
            if random.random() > 0.5:
                x = TF.vflip(x)
                y = TF.vflip(y)
                if self.nuc:
                    n = TF.vflip(n)

            # Random Rotation
            d = random.randint(-180, 180)
            x = TF.rotate(x, d)
            y = TF.rotate(y, d)
            if self.nuc:
                n = TF.rotate(n, d)

        x = TF.to_tensor(x)
        y = TF.to_tensor(y)
        if self.nuc:
            n = TF.to_tensor(n)

        if not self.nuc:
            pair = {"x": x, "y": y, "name": name}
        else:
            pair = {"x": x, "y": y, "n": n, "name": name}

        return pair


if __name__ == '__main__':

    path_to_data = "D:\\4373-T-1_snv02\\"
    train_val_split(path_to_data, 0.9)

    PlayData = DataSet(path_to_data + 'train_names.csv',
                       path_to_data + 'Inputs',
                       path_to_data + 'Outputs',
                       path_to_data + 'Nuclei',
                       transform=CropAndTransform(crop_size=512, augment=True))

    play_data_loader = DataLoader(PlayData, batch_size=1, shuffle=False)

    for i, train_pair in enumerate(play_data_loader):
        print(train_pair["x"].shape)

