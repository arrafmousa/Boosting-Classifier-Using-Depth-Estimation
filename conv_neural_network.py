import argparse
import os
import random

import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from joblib import dump, load
from pathlib import Path
from sklearn import svm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from depth_reconstraction.MiDaS.create_cifar10_depth_dataset import \
    get_data_loaders


def import_rgbd_dataset(data_dir):
    dict = {'apple': 0,
            'ball': 1,
            'banana': 2,
            'bell_pepper': 3,
            'binder': 4,
            'bowl': 5,
            'calculator': 6,
            'camera': 7,
            'cap': 8,
            'cell_phone': 9,
            'cereal_box': 10,
            'coffee_mug': 11,
            'comb': 12,
            'dry_battery': 13,
            'flashlight': 14,
            'food_bag': 15,
            'food_box': 16,
            'food_can': 17,
            'food_cup': 18,
            'food_jar': 19,
            'garlic': 20,
            'glue_stick': 21,
            'greens': 22,
            'hand_towel': 23,
            'instant_noodles': 24,
            'keyboard': 25,
            'kleenex': 26,
            'lemon': 27,
            'lightbulb': 28,
            'lime': 29,
            'marker': 30,
            'mushroom': 31,
            'notebook': 32,
            'onion': 33,
            'orange': 34,
            'peach': 35,
            'pear': 36,
            'pitcher': 37,
            'plate': 38,
            'pliers': 39,
            'potato': 40,
            'rubber_eraser': 41,
            'scissors': 42,
            'shampoo': 43,
            'soda_can': 44,
            'sponge': 45,
            'stapler': 46,
            'tomato': 47,
            'toothbrush': 48,
            'toothpaste': 49,
            'water_bottle': 50}

    obj_list = os.listdir(data_dir)
    rgb_objects = []
    for obj in tqdm(obj_list):
        img_list = os.listdir(data_dir + "/" + obj)
        for img in img_list:
            rot_list = os.listdir(data_dir + "/" + obj + "/" + img)
            num = 10000
            for rot in rot_list:
                num -= 1
                if rot.endswith("_crop.png"):
                    rgb_objects.append([os.path.join(data_dir, obj, img, rot[:-9]), dict[obj]])
                if num <= 0:
                    break
    return rgb_objects


def load_rgbd(data_dir):
    dataset = import_rgbd_dataset(data_dir)
    test_idxes = random.sample(range(0, len(dataset)), int((len(dataset)) * 0.05))
    train_idxes = [x for x in range(0, len(dataset)) if x not in test_idxes]
    test_data = [dataset[test_idx] for test_idx in tqdm(test_idxes)]
    random.shuffle(test_data)
    train_data = [dataset[train_idx] for train_idx in tqdm(train_idxes)]
    random.shuffle(train_data)
    return train_data, test_data


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self, is_depth, is_lstm):
        super().__init__()
        self.conv1 = nn.Conv2d(4 if is_depth else 3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 14, 5)
        self.fc1 = nn.Linear(1050, 512)
        self.fc2 = nn.Linear(512, 200)
        self.fc3 = nn.Linear(200, 10)
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.lsltm = nn.LSTM(
            input_size=5,
            hidden_size=15,
            num_layers=3,
            batch_first=True)
        self.is_lstm = is_lstm

    def forward(self, x):
        x = self.pool(self.conv1(x.squeeze()))
        x = self.pool(self.conv2(x))
        if self.is_lstm:
            x, _ = self.lsltm(x)
        x = torch.flatten(x)  # flatten all dimensions except batch

        svm_in = self.relu(self.fc1(x))
        x = self.relu(self.fc2(svm_in))
        x = self.fc3(x)
        return x, svm_in


def train_and_validate(is_depth, is_lstm,is_svm,num_epochs, cifar_dir=r'C:\Users\USER\Desktop\deep\replication\cifar-10_rgb_format'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"found device {device}")
    ## FOR Object-RGBD training
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #
    # classes = ['apple', 'ball', 'banana', 'bell_pepper', 'binder', 'bowl', 'calculator', 'camera', 'cap', 'cell_phone',
    #            'cereal_box', 'coffee_mug', 'comb', 'dry_battery', 'flashlight', 'food_bag', 'food_box', 'food_can',
    #            'food_cup', 'food_jar', 'garlic', 'glue_stick', 'greens', 'hand_towel', 'instant_noodles', 'keyboard',
    #            'kleenex', 'lemon', 'lightbulb', 'lime', 'marker', 'mushroom', 'notebook', 'onion', 'orange', 'peach',
    #            'pear', 'pitcher', 'plate', 'pliers', 'potato', 'rubber_eraser', 'scissors', 'shampoo', 'soda_can',
    #            'sponge', 'stapler', 'tomato', 'toothbrush', 'toothpaste', 'water_bottle']
    #
    # train_data, test_data = load_rgbd(
    #     r'C:\Users\USER\Desktop\deep\Deep-Belief-Networks-in-PyTorch\dataset\rgbd-dataset')
    #
    # # get some random training images
    # dataiter = iter(train_data)
    # images, labels = next(dataiter)
    #
    # # show images
    # convert_tensor = transforms.ToTensor()
    # imshow(torchvision.utils.make_grid(convert_tensor(PIL.Image.open(images + "_crop.png"))))
    # # print labels
    # print(classes[labels])
    #
    if not is_depth:
        transform = transforms.Compose(
            [transforms.ToTensor(),  # uint8 values in [0, 255] -> float tensor with vlaues [0, 1]
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(),  # uint8 values in [0, 255] -> float tensor with vlaues [0, 1]
             transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
             ])

    cifar10_rgbd_output_dir = Path(
        r'C:\Users\USER\Desktop\deep\replication\cifar-10_rgb_format') if cifar_dir is None else Path(cifar_dir)
    # get data loaders
    train_size = float('inf')
    trainloader, testloader, valloader, labels_classes = \
        get_data_loaders(cifar10_rgbd_output_dir, 1, train_size, is_depth, transform)

    net = Net(is_depth, is_lstm).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    svm_acc = []
    model_acc = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        X, Y = [], []
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, _ = net(inputs.to(device))
            loss = criterion(outputs.view(1, 10), labels.to(device))
            loss.backward()
            optimizer.step()

        torch.save(net.state_dict(), f"fc2_conv_lstm_depth_model_epoch_{epoch}")

        # net.load_state_dict(torch.load(rf"fc2_conv_model_epoch_{epoch}"))
        # val_loss = 0
        network_val_acc = []
        for i, data in enumerate(testloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward + backward + optimize
            outputs, _ = net(inputs.to(device))
            network_val_acc.append(1 if outputs.argmax().cpu() == labels else 0)
        print(f"\t \t network validation acc for {epoch} is {sum(network_val_acc) / len(network_val_acc)}")
        model_acc.append(sum(network_val_acc) / len(network_val_acc))

        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            X.append(net(inputs.to(device))[1].cpu().detach().numpy())
            Y.append(labels[0].item())

        if is_svm:
            # print("fitting SVM")
            clf = svm.SVC(C=2, kernel='poly', degree=1)  # verbose=True
            clf.fit(X, Y)
            print("done fitting")
            dump(clf, f"fc2_conv_lstm_depth_svm_model_epoch_{epoch}")
            # clf = joblib.load(f"fc2_conv_svm_model_epoch_{epoch}")

        svm_val_acc = []
        for i, data in enumerate(testloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if is_svm:
                pred = clf.predict([net(inputs.to(device))[1].cpu().detach().numpy()])
            else:
                outputs, _ = net(inputs.to(device))
                pred = outputs.argmax().cpu()
            svm_val_acc.append(1 if pred.item() == labels.item() else 0)
        print(f"\t \t svm validation acc for {epoch} is {sum(svm_val_acc) / len(svm_val_acc)}")
        svm_acc.append(sum(svm_val_acc) / len(svm_val_acc))

    plt.plot(svm_acc, color='r', label='svm_depth_acc')
    plt.plot(model_acc, color='g', label='model_depth_acc')
    plt.legend()
    plt.show()


def plot_results():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [transforms.ToTensor(),  # uint8 values in [0, 255] -> float tensor with vlaues [0, 1]
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
    transform_depth = transforms.Compose(
        [transforms.ToTensor(),  # uint8 values in [0, 255] -> float tensor with vlaues [0, 1]
         transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
         ])

    cifar10_rgbd_output_dir = Path(
        r'C:\Users\USER\Desktop\deep\replication\cifar-10_rgb_format')
    # get data loaders
    train_size = float('inf')
    _, depth_testloader, _, _ = \
        get_data_loaders(cifar10_rgbd_output_dir, 1, train_size, True, transform_depth)
    _, testloader, _, _ = \
        get_data_loaders(cifar10_rgbd_output_dir, 1, train_size, False, transform)
    svm_acc = []
    svm_depth_acc = []
    network_acc = []
    network_depth_acc = []
    network_lstm_acc = []
    network_lstm_depth_acc = []
    for epoch in range(30):
        clf_depth = joblib.load(f"fc2_conv_depth_svm_model_epoch_{epoch}")
        clf = joblib.load(f"fc2_conv_svm_model_epoch_{epoch}")
        net_depth = Net(True, False).to(device)
        net = Net(False, False).to(device)
        net_lstm_depth = Net(True, True).to(device)
        net_lstm = Net(False, True).to(device)
        net.load_state_dict(torch.load(rf"fc2_conv_model_epoch_{epoch}"))
        net_depth.load_state_dict(torch.load(rf"fc2_conv_depth_model_epoch_{epoch}"))
        net_lstm.load_state_dict(torch.load(rf"fc2_conv_lstm_model_epoch_{epoch}"))
        net_lstm_depth.load_state_dict(torch.load(rf"fc2_conv_lstm_depth_model_epoch_{epoch}"))

        results = []
        for i, data in enumerate(testloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            pred = clf.predict([net(inputs.to(device))[1].cpu().detach().numpy()])
            results.append(1 if pred.item() == labels.item() else 0)
        print(f"\t \t svm validation acc for {epoch} is {sum(results) / len(results)}")
        svm_acc.append(sum(results) / len(results))

        results = []
        for i, data in enumerate(depth_testloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            pred = clf_depth.predict([net_depth(inputs.to(device))[1].cpu().detach().numpy()])
            results.append(1 if pred.item() == labels.item() else 0)
        print(f"\t \t svm depth validation acc for {epoch} is {sum(results) / len(results)}")
        svm_depth_acc.append(sum(results) / len(results))

        results = []
        for i, data in enumerate(testloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward + backward + optimize
            outputs, _ = net(inputs.to(device))
            results.append(1 if outputs.argmax().cpu() == labels else 0)
        print(f"\t \t network validation acc for {epoch} is {sum(results) / len(results)}")
        network_acc.append(sum(results) / len(results))

        results = []
        for i, data in enumerate(depth_testloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward + backward + optimize
            outputs, _ = net_depth(inputs.to(device))
            results.append(1 if outputs.argmax().cpu() == labels else 0)
        print(f"\t \t network validation acc for {epoch} is {sum(results) / len(results)}")
        network_depth_acc.append(sum(results) / len(results))

        results = []
        for i, data in enumerate(depth_testloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward + backward + optimize
            outputs, _ = net_lstm_depth(inputs.to(device))
            results.append(1 if outputs.argmax().cpu() == labels else 0)
        print(f"\t \t network validation acc for {epoch} is {sum(results) / len(results)}")
        network_lstm_depth_acc.append(sum(results) / len(results))

        results = []
        for i, data in enumerate(testloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward + backward + optimize
            outputs, _ = net_lstm(inputs.to(device))
            results.append(1 if outputs.argmax().cpu() == labels else 0)
        print(f"\t \t network validation acc for {epoch} is {sum(results) / len(results)}")
        network_lstm_acc.append(sum(results) / len(results))


    plt.plot(svm_acc, color='r', label='svm accuracy')
    plt.plot(svm_depth_acc, color='g', label='svm with depth accuracy')
    plt.legend()
    plt.show()

    plt.plot(network_acc, color='r', label='network accuracy')
    plt.plot(network_depth_acc, color='g', label='network with depth accuracy')
    plt.legend()
    plt.show()

    plt.plot(svm_acc, color='r', label='svm accuracy')
    plt.plot(svm_depth_acc, color='g', label='svm with depth accuracy')
    plt.plot(network_acc, color='b', label='network accuracy')
    plt.plot(network_depth_acc, color='c', label='network with depth accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_lstm", type=bool)
    parser.add_argument("--is_depth", type=bool)
    parser.add_argument("--cifar_dir", type=bool)
    parser.add_argument("--is_svm", type=bool)
    parser.add_argument("--num_epochs", type=int)
    parser.parse_args()
    args = parser.parse_args()
    train_and_validate(args.is_depth, args.is_lstm, args.is_svm, args.num_epochs)
    plot_results()
