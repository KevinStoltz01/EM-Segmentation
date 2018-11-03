
import sys
import os
import numpy as np
import pandas as pd
from PIL import Image
from UNet_V0 import UNet
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from metrics import Metrics, MetricsinMemory, metrics_grid_search, metrics_in_memory_grid_search
import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

dataset = "DATASET NAME"
img_size = 512
dataset = str(dataset)
experiment = "EXPERIMENT NAME"
random_crops = True
semi_supervised = True
save_predictions = False
num_experiments = 10
num_epochs = 30
batch_size = 1
learning_rate = 1e-3
if semi_supervised:
    test_size1 = 0.9
    test_size2 = 0.5
if not semi_supervised:
    test_size1 = 0.2
    test_size2 = 0.25
wpix_threshold = 0.25
keep_prob = 0.1


def main():
    # name the experiment and set file paths
    project_folder = PATH
    saved_models = project_folder + PATH
    path_to_data = project_folder + PATH
    path_to_predictions = project_folder + PATH
    path_to_training_logs = project_folder + PATH
    path_to_metrics_logs = project_folder + PATH
   

# Start Experiments ====================================================================================================
# ======================================================================================================================
# ======================================================================================================================

    # Training Loop ====================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================

    training_csv_logger = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    metrics_csv_logger = {"accuracy": [],
                          "miou": [],
                          "% white pixels full images": [],
                          "% white pixels crops": [],
                          "keep_prob": []}
    for k in range(num_experiments):
        metrics_csv_logger["keep_prob"].append(keep_prob)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Preparing the data set
        white_pixels_full = train_val_split(path_to_data, 0.9)
        metrics_csv_logger["% white pixels full images"].append(white_pixels_full / 100)
        train = DataSet(path_to_data + experiment + 'train_names.csv',
                        path_to_data + 'Inputs',
                        path_to_data + 'Outputs',
                        path_to_data + 'Nuclei',
                        transform=CropAndTransform(crop_size=512, augment=True))
        val = DataSet(path_to_data + experiment + 'train_names.csv',
                      path_to_data + 'Inputs',
                      path_to_data + 'Outputs',
                      path_to_data + 'Nuclei',
                      transform=CropAndTransform(crop_size=512))

        train_data_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        val_data_loader = DataLoader(val, batch_size=1, shuffle=False)

        # Initializing model, optimizer, and loss function
        unet = UNet().to(device)
        if unet.conv10.cross_entropy:
            if unet.conv10.mode == "single":
                loss_fxn = nn.BCEWithLogitsLoss()
            if unet.conv10.mode == "multi":
                loss_fxn = nn.CrossEntropyLoss()
        elif not unet.conv10.cross_entropy:
            loss_fxn = nn.MSELoss()
        optimizer = optim.Adam(unet.parameters(), lr=learning_rate)

        ones = torch.ones([1, 1, img_size, img_size]).to(device)
        zeros = torch.zeros([1, 1, img_size, img_size]).to(device)

        # training loop
        best_val_loss = 1
        white_pixels_crops_experiment = 0
        for epoch in range(num_epochs):
            total_train_loss = 0
            total_train_acc = 0
            unet.train()
            white_pixels_crops_epoch = 0
            for i, train_pair in enumerate(train_data_loader):
                # get input and gt
                x = train_pair["x"].to(device)
                y = train_pair["y"].to(device)
                y_array = y.detach().cpu().squeeze()
                # calculate the % white pixels in gt image
                y_array = y_array.numpy()
                white_pix = np.sum(y_array)
                white_pixels_crops_epoch += (white_pix / (y_array.shape[0] * y_array.shape[1]))
                avg_white_pixels_crops_epoch = white_pixels_crops_epoch / (i + 1)
                # run forward and backprop
                optimizer.zero_grad()
                yhat = unet(x)
                loss = loss_fxn(yhat, y)
                yhat = torch.where(yhat > 0.5, ones, zeros)
                loss.backward()
                optimizer.step()
                # calculate training metrics
                total_train_loss += loss.item()
                avg_train_loss = total_train_loss / (i + 1)
                acc = torch.eq(y, yhat).sum().item()
                acc = acc / (img_size * img_size)
                total_train_acc += acc
                avg_train_acc = total_train_acc / (i + 1)

                print('experiment [{}/{}], epoch [{}/{}], batch [{}/{}], loss: {:02.4f}, acc: {:.4f}'.format(
                    k + 1, num_experiments,
                    epoch + 1, num_epochs,
                    i + 1, int(len(train) / batch_size),
                    avg_train_loss, avg_train_acc))
                sys.stdout.flush()

                if i == int(len(train) / batch_size) - 1:
                    training_csv_logger["train_loss"].append(avg_train_loss)
                    training_csv_logger["train_acc"].append(avg_train_acc)
                    white_pixels_crops_experiment += avg_white_pixels_crops_epoch
                    avg_white_pixels_crops_experiment = white_pixels_crops_experiment / (epoch + 1)

            unet.eval()
            total_val_loss = 0
            avg_val_loss = 0
            total_val_acc = 0
            for j, val_pair in enumerate(val_data_loader):
                x = val_pair["x"].to(device)
                y = val_pair["y"].to(device)
                yhat = unet(x)
                loss = loss_fxn(yhat, y)
                total_val_loss += loss.item()
                yhat = torch.where(yhat > 0.5, ones, zeros)
                avg_val_loss = total_val_loss / (j + 1)
                acc = torch.eq(y, yhat).sum().item()
                acc = acc / (img_size * img_size)
                total_val_acc += acc
                avg_val_acc = total_val_acc / (j + 1)

                if j == int(len(val) / batch_size) - 1:
                    training_csv_logger["val_loss"].append(avg_val_loss)
                    training_csv_logger["val_acc"].append(avg_val_acc)

            if avg_val_loss < best_val_loss:
                print('validation loss improved from {:.4f}, to {:.4f}. Saving model'.format(
                    best_val_loss, avg_val_loss))
                sys.stdout.flush()
                torch.save(unet.state_dict(), saved_models)
                best_val_loss = avg_val_loss

            else:
                print('validation loss did not improve')
                sys.stdout.flush()

            training_csv_logger["epoch"].append(epoch + 1)
            training_log = pd.DataFrame(training_csv_logger)
            training_log.to_csv(path_to_training_logs)

    # Inference ========================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
        if save_predictions:
            if not os.path.isdir(path_to_predictions):
                os.mkdir(path_to_predictions)

        # Prepare data loading tools
        if semi_supervised:
            test = DataSet(path_to_data + "Tiled\\" + 'names.csv',
                           path_to_data + "Tiled\\" + 'Inputs',
                           path_to_data + "Tiled\\" + 'Outputs',
                           path_to_data + "Tiled\\" + 'Nuclei',
                           transform=CropAndTransform(Random_Crops=False))
        else:
            test = DataSet(path_to_data + experiment + 'test_names.csv',
                           path_to_data + 'Inputs',
                           path_to_data + 'Outputs',
                           path_to_data + 'Nuclei',
                           transform=CropAndTransform(crop_size=512))

        test_data_loader = DataLoader(test, batch_size=1, shuffle=False)

        # Set device to 'Cuda' if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load network parameters and set mode to eval
        unet = UNet().to(device)
        unet.load_state_dict(torch.load(saved_models))
        unet.eval()

        number_of_images = 0
        Accuracy = 0
        mIou = 0
        for Input in test_data_loader:
            x = Input["x"].to(device)
            name = Input["name"]
            yhat = unet(x)
            yhat = yhat.detach().cpu().squeeze()
            yhat = yhat.numpy()
            y = Input["y"].detach().cpu().squeeze()
            y = y.numpy()

            acc, miou = MetricsinMemory(y, yhat, 2, 0.5)
            Accuracy += acc
            mIou += miou

            if save_predictions:
                yhat = Image.fromarray(yhat)
                yhat.save(path_to_predictions + str(name) + ".tiff")

            number_of_images += 1

        avg_miou = mIou / number_of_images
        avg_accuracy = Accuracy / number_of_images

        metrics_csv_logger["accuracy"].append(avg_accuracy)
        metrics_csv_logger["miou"].append(avg_miou)
        metrics_csv_logger["% white pixels crops"].append(avg_white_pixels_crops_experiment)

        print("accuraacy - {}".format(metrics_csv_logger["accuracy"]))
        print("miou - {}".format(metrics_csv_logger["miou"]))
        print("% white pixels full images - {}".format(metrics_csv_logger["% white pixels full images"]))
        print("% white pixels crops - {}".format(metrics_csv_logger["% white pixels crops"]))

        summary = pd.DataFrame(metrics_csv_logger)
        summary.to_csv(path_to_metrics_logs)

        if num_experiments > 1:
            print("Beginning next experiment...")
            sys.stdout.flush()

    os.remove(path_to_data + experiment + 'train_names.csv')
    os.remove(path_to_data + experiment + 'val_names.csv')


# Data Loader =========================================================================================================
# =====================================================================================================================
# =====================================================================================================================
def train_val_split(directory, white_pix_threshold=0.95):
    names = pd.read_csv(directory + 'names.csv')
    names = names.iloc[:, 1].tolist()

    train, valtest = train_test_split(names, test_size=test_size1, random_state=random.randint(0, 100))
    val, test = train_test_split(valtest, test_size=test_size2, random_state=random.randint(0, 100))

    df_train = pd.DataFrame(train)
    df_val = pd.DataFrame(val)
    df_test = pd.DataFrame(test)
    df_train.to_csv(directory + experiment + "train_names.csv")
    df_val.to_csv(directory + experiment + "val_names.csv")
    df_test.to_csv(directory + experiment + "test_names.csv")

    train_names = [glob.glob(directory + "Outputs\\" + name) for name in train]

    number_of_images = 0
    percent_white_pix = 0
    for i, name in enumerate(train_names):
        print("checking image [{}/{}] for white pixel content".format(i + 1, len(train_names)))
        im = Image.open(train_names[i][0])
        im_array = np.array(im)
        im_array_sum = np.sum(np.where(im_array > 0, 1, 0))
        percent_white_pix += (im_array_sum / (im_array.shape[0] * im_array.shape[1])) * 100
        avg_percent_white_pix = percent_white_pix / (number_of_images + 1)
        number_of_images += 1

    if avg_percent_white_pix < white_pix_threshold:
        print("Searching for images with Nucleoli")
        train_val_split(directory, white_pix_threshold)
    else:
        print("average percentage of white pixels - {}".format(avg_percent_white_pix))

    return avg_percent_white_pix


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
    def __init__(self, crop_size=512,
                 augment=False,
                 nuc=False,
                 Wpix_Threshold=wpix_threshold,
                 Keep_Prob=keep_prob,
                 Random_Crops=True):
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        elif isinstance(crop_size, tuple):
            self.crop_size = crop_size
        else:
            raise TypeError("Variable crop_size must be type int or tuple")
        self.augment = augment
        self.nuc = nuc
        self.Wpix_Threshold = Wpix_Threshold
        self.Keep_Prob = Keep_Prob
        self.Random_Crops = Random_Crops

    def __call__(self, images):
        name = images["name"]
        x = images["x"]
        y = images["y"]
        if self.nuc:
            n = images["n"]

        #  check to see if y contains nucleoli and reset the process if it does not
        if self.Random_Crops:
            wpix = 0
            while wpix < self.Wpix_Threshold:
                # Random crops
                i, j, h, w = transforms.RandomCrop.get_params(x, output_size=self.crop_size)
                x = TF.crop(x, i, j, h, w)
                y = TF.crop(y, i, j, h, w)
                y_array = np.array(y) / 255
                if self.nuc:
                    n = TF.crop(n, i, j, h, w)
                wpix = np.sum(y_array) / (y_array.shape[0] * y_array.shape[1])
                if wpix < self.Wpix_Threshold:
                    local_keep_prob = random.random()
                    if local_keep_prob < self.Keep_Prob:
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


if __name__ == "__main__":
    main()
