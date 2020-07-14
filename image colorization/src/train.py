import os
import numpy as np
import subprocess
import torch

from decimal import Decimal
from utils import read_image, cvt2Lab, upsample, cvt2rgb


def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]


class ConvNet(torch.nn.Module):
    """convolutional network that performs image colorization"""
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(kernel_size=2))

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.MaxPool2d(kernel_size=2))

        self.layer3 = torch.nn.Conv2d(8, 2, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# IMAGE PATHS
TRAIN_IMAGENAME_PATH    = "train.txt"
VALID_IMAGENAME_PATH    = "valid.txt"

GRAY_IMAGE_PATH         = "../data/gray/"
COLOR_64_IMAGE_PATH     = "../data/color_64/"

MODEL_PATH              = "../model/image_colorization_model.pt"

### MAIN ### MAIN ### MAIN ### MAIN ### MAIN ### MAIN ### MAIN ### MAIN ### MAIN ### MAIN ### MAIN ### MAIN ### MAIN ###
def main():
    
    # LOAD TRAINING IMAGE NAMES
    with open(TRAIN_IMAGENAME_PATH, 'r') as infile:
        global train_imagename
        train_imagename = [line.strip() for line in infile]

    # LOAD VALIDATION IMAGE NAMES
    with open(VALID_IMAGENAME_PATH, 'r') as infile:
        global valid_imagename
        valid_imagename = [line.strip() for line in infile]

    print("-> image names are loaded")
    print()

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #

    # LOAD GRAY IMAGES
    gray_images = list()
    for gray_imagename in os.listdir(GRAY_IMAGE_PATH):
        gray_image, s   = read_image(GRAY_IMAGE_PATH+gray_imagename)
        gray_image      = (gray_imagename, cvt2Lab(gray_image)[0])
        gray_images.append(gray_image)
    gray_images = sorted(gray_images, key=lambda x: x[0])
    print("-> gray images are loded")

    # SPLIT GRAY IMAGES TO TRAIN AND VALIDATION SETS
    x_train, x_valid = np.empty([1, 1, 256, 256]), np.empty([1, 1, 256, 256])
    for gray_imagename, gray_image in gray_images:
        if gray_imagename in train_imagename:
            x_train = np.concatenate([x_train, np.reshape(gray_image, (1,) + x_train.shape[1:])])
        if gray_imagename in valid_imagename:
            x_valid = np.concatenate([x_valid, np.reshape(gray_image, (1,) + x_valid.shape[1:])])
    x_train, x_valid = x_train[1:], x_valid[1:]
    print("-> gray images are splitted to datasets")
    print()

    # SAVE TRAINING AND VALIDATION FEATURES
    #np.save("x_train.npy", x_train)
    #print("-> x_train is saved")
    #np.save("x_valid.npy", x_valid)
    #print("-> x_valid is saved")
    #print()

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #

    # LOAD 64X64 COLOR IMAGES
    color64_images = list()
    for color64_imagename in os.listdir(COLOR_64_IMAGE_PATH):
        color64_image, s    = read_image(COLOR_64_IMAGE_PATH+color64_imagename, training=True)
        color64_image       = (color64_imagename, cvt2Lab(color64_image)[1])
        color64_images.append(color64_image)
    color64_images  = sorted(color64_images, key=lambda x:x[0])
    print("-> 64x64 color images are loded")

    # SPLIT 64x64 COLOR IMAGES TO TRAIN AND VALIDATION SETS
    y_train, y_valid    = np.empty([1, 64, 64, 2]), np.empty([1, 64, 64, 2])
    for color64_imagename, color64_image in color64_images:
        if color64_imagename in train_imagename:
            y_train = np.concatenate([y_train, np.expand_dims(color64_image, axis=0)])
        if color64_imagename in valid_imagename:
            y_valid = np.concatenate([y_valid, np.expand_dims(color64_image, axis=0)])
    y_train, y_valid = np.rollaxis(y_train[1:], 3, 1), np.rollaxis(y_valid[1:], 3, 1)
    print("-> 64x64 color images are splitted to datasets")
    print()

    # SAVE TRAINING AND VALIDATION LABELS
    #np.save("y_train.npy", y_train)
    #print("-> y_train is saved")
    #np.save("y_valid.npy", y_valid)
    #print("-> y_valid is saved")
    #print()

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    
    # LOAD TRAINING AND TEST SAMPLES
    #x_train = np.load('x_train.npy')
    #x_valid = np.load('x_valid.npy')
    #y_train = np.load('y_train.npy')
    #y_valid = np.load('y_valid.npy')

    # DISPLAY DATASET SIZE
    train_size  = x_train.shape[0]
    valid_size  = x_valid.shape[0]
    print("-> train Size : %d" % train_size)
    print("-> valid Size : %d" % valid_size)
    print()

    # SET TRAINING PARAMETERS
    BATCH_SIZE  = 50
    EPOCH       = 250

    model       = ConvNet().cuda()
    loss_fn     = torch.nn.MSELoss()
    optimizer   = torch.optim.RMSprop(model.parameters(), lr=1e-4)

    ### TRAINING ### TRAINING ### TRAINING ### TRAINING ### TRAINING ### TRAINING ### TRAINING ### TRAINING ### TRAINING ###
    for epoch in range(EPOCH):
        running_train_loss = .0
        running_valid_loss = .0
        for i in range(0, train_size, BATCH_SIZE):
            trainX, trainY = torch.autograd.Variable(torch.from_numpy(x_train[i:i + BATCH_SIZE]).float().cuda(), requires_grad=False),\
                             torch.autograd.Variable(torch.from_numpy(y_train[i:i + BATCH_SIZE]).float().cuda(), requires_grad=False)

            optimizer.zero_grad()

            train_output    = model(trainX)
            train_loss      = loss_fn(train_output, trainY)

            train_loss.backward()
            optimizer.step()

            running_train_loss += train_loss.data[0]

        for i in range(0, valid_size, BATCH_SIZE):
            validX, validY = torch.autograd.Variable(torch.from_numpy(x_valid[i:i + BATCH_SIZE]).float().cuda(), requires_grad=False),\
                             torch.autograd.Variable(torch.from_numpy(y_valid[i:i + BATCH_SIZE]).float().cuda(), requires_grad=False)

            valid_output = model(validX)
            valid_loss = loss_fn(valid_output, validY)

            running_valid_loss += valid_loss.data[0]

        print("%d\ttrain loss : %s\t%s" % (epoch+1, str(format(running_train_loss / (x_train.shape[0] / BATCH_SIZE), '.8g')), str(format(running_valid_loss / (x_valid.shape[0] / BATCH_SIZE), '.8g'))))

    # GET VALIDATION PREDICTIONS
    # Shape -> (100, 2, 64, 64)
    pred_valid = np.vstack([model(torch.autograd.Variable(torch.from_numpy(x_valid[i:i + BATCH_SIZE]).float().cuda(),
                                                          requires_grad=False)).cpu().data.numpy() \
                            for i in range(0, valid_size, BATCH_SIZE)])

    # ADJUST VALIDATION PREDICTION DIMENSIONS
    # Shape -> (100, 64, 64, 2)
    pred_valid = np.transpose(pred_valid, (0, 2, 3, 1))

    # UPSAMPLE VALIDATION PREDICTIONS
    # Shape -> (100, 256, 256, 2)
    pred_valid = np.vstack([np.expand_dims(upsample(pred.astype(np.float)), axis=0) \
                            for pred in pred_valid])

    # INSERT LIGHT CHANNEL TO VALIDATION PREDICTIONS
    # Shape -> (100, 256, 256, 3)
    pred_valid = np.vstack([np.expand_dims(np.insert(pred_valid[i], 0, x_valid[i], axis=2), axis=0) \
                            for i in range(len(pred_valid))])

    # CONVERT VALIDATION PREDICTIONS TO RGB IMAGES
    # Shape -> (100, 256, 256, 3)
    pred_valid = np.vstack([np.expand_dims((cvt2rgb(pred) * 255.).astype(np.uint8), axis=0) \
                            for pred in pred_valid])

    np.save('validation_estimations.npy', pred_valid)

    # DISPLAY VALIDATION ACCURACY
    print("Validation acc: ")
    subprocess.call(["python3", "evaluate.py", "validation_estimations.npy", "valid.txt"])
    print()

    try:
        os.remove('validation_estimations.npy')
    except:
        pass

    # SAVE PYTORCH MODEL
    torch.save(model.state_dict(), MODEL_PATH)
    print("-> image colorization model is saved to %s" % MODEL_PATH)

    return


if __name__ == '__main__':
    main()
