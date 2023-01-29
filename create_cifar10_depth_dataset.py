from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np
from enum import Enum
import cv2
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import argparse

from load_cifar10_dataset import LoadCifar10
from create_dataset_from_dir import create_dataset, ImagToPlot, plot_images


def save_image(output_path, image):
    print("saving image: ", str(output_path))
    plt.imsave(output_path, image, cmap='gray')


def save_images_to_dir(output_path, x, y):
    os.makedirs(output_path, exist_ok=True)

    for i, img in enumerate(tqdm(x, desc=save_images_to_dir.__name__)):
        image_path = output_path/(str(i)+".png")

        if not os.path.isfile(image_path):
            save_image(image_path, img)

def read_raw_cifar_images_and_save_to_folder(cifar10_raw_data_dir, destination_dir):
    x_train, y_train, x_val, y_val, x_test, y_test, label_names = LoadCifar10().get_CIFAR10_data(cifar10_raw_data_dir)

    print('Train data shape: ', x_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', x_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', x_test.shape)
    print('Test labels shape: ', y_test.shape)

    train_output_path_train = destination_dir / "train"
    save_images_to_dir(train_output_path_train, x_train, y_train)
    with open(destination_dir / "labels_train.txt", 'w') as output:
        for label in y_train:
            output.write(str(label)+"\n")

    val_output_path_train = destination_dir / "val"
    save_images_to_dir(val_output_path_train, x_val, y_val)
    with open(destination_dir / "labels_val.txt", 'w') as output:
        for label in y_val:
            output.write(str(label)+"\n")

    test_output_path_train = destination_dir / "test"
    save_images_to_dir(test_output_path_train, x_test, y_test)
    with open(destination_dir / "labels_test.txt", 'w') as output:
        for label in y_test:
            output.write(str(label)+"\n")

    with open(destination_dir / "label_names.txt", 'w') as output:
        for label_name in label_names:
            output.write(str(label_name)+"\n")

    return train_output_path_train, val_output_path_train, test_output_path_train

def create_depth_images(train_output_path_train, val_output_path_train, test_output_path_train):
    create_dataset(train_output_path_train)
    create_dataset(val_output_path_train)
    create_dataset(test_output_path_train)

def create_rgbd_dataset(cifar10_raw_data_dir, destination_dir):
    train_output_path_train, val_output_path_train, test_output_path_train = \
        read_raw_cifar_images_and_save_to_folder(cifar10_raw_data_dir=cifar10_raw_data_dir,
                                                 destination_dir=destination_dir)

    create_depth_images(train_output_path_train, val_output_path_train, test_output_path_train)


class DataType(Enum):
    train = 1
    test = 2
    val = 3

def get_all_images_paths_in_dir(dir_path, image_type=".png"):
    img_files = os.listdir(dir_path)
    if image_type is not None:
        img_files = list(filter(lambda x: image_type in x, img_files))
    img_files.sort(key=lambda x: int(x.split(".png")[0]))
    return [Path(dir_path) / Path(file_name) for file_name in img_files]


class Cifar10Dataset(Dataset):
    def __init__(self, cifar10_rgbd_dir, data_type, add_depth, max_size, transform=None):
        images_dir = cifar10_rgbd_dir / data_type.name
        depth_images_dir = cifar10_rgbd_dir / (data_type.name + "_depth")
        labels_file = cifar10_rgbd_dir / ("labels_" + data_type.name + ".txt")
        labels_classes_file = cifar10_rgbd_dir / "label_names.txt"

        self.max_size = max_size
        self.add_depth = add_depth

        self.images = self._read_images(images_dir, depth_images_dir, self.max_size)
        self.labels = self._read_labels(labels_file, self.max_size)
        self.labels_classes = self._read_labels_classes(labels_classes_file)

        self.transform = transform

        assert len(self.images) == len(self.labels)

    def _read_file_to_list(self, file_path, convert_to_int=False):
        with open(file_path, 'r') as f:
            data = f.read()
            data_list = data.replace('\n', ' ').split(" ")[:-1]
            if convert_to_int:
                data_list = [int(label) for label in data_list]
            return data_list

    def _read_images(self, images_dir, depth_images_dir, max_size):
        images_paths = get_all_images_paths_in_dir(images_dir)
        images_paths = images_paths[: min(max_size, len(images_paths))]

        img_shape = LoadCifar10().images_shape
        n_dims = img_shape[2]
        if self.add_depth:
            n_dims += 1

        images = np.empty([len(images_paths), img_shape[0], img_shape[1], n_dims], dtype=np.float)

        for i, img_path in enumerate(tqdm(images_paths, desc=self._read_images.__name__)):
            if i == max_size:
                break

            rgb_img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            rgb_img = rgb_img / 255

            if self.add_depth:
                img_name = os.path.basename(img_path)
                depth_image_path = depth_images_dir/img_name
                depth_img = cv2.imread(str(depth_image_path), cv2.IMREAD_GRAYSCALE)

                normalized_depth = True
                if normalized_depth:
                    depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
                else:
                    depth_img = depth_img / 255

                images[i, :, :, 0:3] = rgb_img
                images[i, :, :, 3] = depth_img
            else:
                images[i, :, :, :] = rgb_img

        return images

    def _read_labels(self, labels_file, max_size):
        labels = self._read_file_to_list(file_path=labels_file, convert_to_int=True)
        return labels[0:min(len(labels), max_size)]

    def _read_labels_classes(self, labels_classes_file):
         return self._read_file_to_list(file_path=labels_classes_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image.type(torch.float32), label


def get_data_loaders(cifar10_rgbd_output_dir, batch_size, max_size, add_depth, transform):
    max_size = min(60_000, max_size)

    max_test = min(10_000, int(max_size * 0.25))
    max_val = min(1_000, int(max_size * 0.15))
    max_train = max_size - max_val - max_test

    trainset = Cifar10Dataset(cifar10_rgbd_dir=cifar10_rgbd_output_dir,
                              data_type=DataType.train,
                              add_depth=add_depth,
                              max_size=max_train,
                              transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = Cifar10Dataset(cifar10_rgbd_dir=cifar10_rgbd_output_dir,
                             data_type=DataType.test,
                             add_depth=add_depth,
                             max_size=max_test,
                             transform=transform)
    testsetloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    valset = Cifar10Dataset(cifar10_rgbd_dir=cifar10_rgbd_output_dir,
                            data_type=DataType.val,
                            add_depth=add_depth,
                            max_size=max_val,
                            transform=transform)
    valsetloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    return trainloader, testsetloader, valsetloader, valset.labels_classes


def unnormalize_image(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if len(img.shape) == 3:
        # in case of rgb image
        return np.transpose(npimg, (1, 2, 0))
    else:
        # in case or 2d depth image
        return npimg


def plot_dataloader_images(data_loader, labels_classes, num_of_images_to_plot=4):
    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    if images.shape[1] == 4:
        images_to_plot = []
        for i, image in enumerate(images):
            if i == num_of_images_to_plot:
                break
            rgb_image = unnormalize_image(image[0:3, :, :])
            depth_image = unnormalize_image(image[3, :, :])
            # original_image = np.moveaxis(original_image.numpy(), 0, -1)
            images_to_plot.append(ImagToPlot(labels_classes[labels[i]], rgb_image, depth_image))

        plot_images(images_to_plot)
    else:
        img = unnormalize_image(torchvision.utils.make_grid(images))
        plt.imshow(img)
        plt.show()
    print(' '.join(f'{labels_classes[labels[j]]:5s}' for j in range(num_of_images_to_plot)))


def main(prepare_dataset, cifar10_dir, cifar10_RGBD_output_dir):
    cifar10_dir = Path(cifar10_dir) #Path(r'C:\Users\ch3nk\Desktop\technion\deep_learning\deep_learning_046211_hw\final_project\datasets\cifar-10\cifar-10-batches-py')
    cifar10_rgbd_output_dir = Path(cifar10_RGBD_output_dir) #Path(r'C:\Users\ch3nk\Desktop\technion\deep_learning\deep_learning_046211_hw\final_project\datasets\cifar-10_rgb_format')
    if prepare_dataset:
        create_rgbd_dataset(cifar10_raw_data_dir=cifar10_dir, destination_dir=cifar10_rgbd_output_dir)

    add_depth = True
    if not add_depth:
        transform = transforms.Compose(
            [transforms.ToTensor(),  # uint8 values in [0, 255] -> float tensor with vlaues [0, 1]
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(),  # uint8 values in [0, 255] -> float tensor with vlaues [0, 1]
             transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
             ])

    # get data loaders
    batch_size = 5
    train_size = 100

    trainloader, testsetloader, valsetloader, labels_classes = \
        get_data_loaders(cifar10_rgbd_output_dir, batch_size, train_size, add_depth, transform)

    plot_dataloader_images(trainloader, labels_classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare_dataset', action='store_true')
    parser.add_argument('--cifar10_dir',
            help='the folder with the extracted files from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            required=True)
    parser.add_argument('--cifar10_RGBD_output_dir', required=True)
    args = parser.parse_args()
    main(args.prepare_dataset, args.cifar10_dir, args.cifar10_RGBD_output_dir)
