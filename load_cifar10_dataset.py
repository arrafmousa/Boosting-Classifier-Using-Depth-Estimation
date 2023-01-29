import numpy as np
from six.moves import cPickle as pickle
import os
import platform
import matplotlib.pyplot as plt
import argparse

class LoadCifar10():
    def __init__(self):
        self.classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        img_rows, img_cols = 32, 32
        self.images_shape = (img_rows, img_cols, 3)

    def _load_pickle(self, f):
        version = platform.python_version_tuple()
        if version[0] == '2':
            return pickle.load(f)
        elif version[0] == '3':
            return pickle.load(f, encoding='latin1')
        raise ValueError("invalid python version: {}".format(version))


    def _load_CIFAR_batch(self, filename):
        """ load single batch of cifar """
        with open(filename, 'rb') as f:
            datadict = self._load_pickle(f)
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000,3072)
            Y = np.array(Y)
            return X, Y


    def _load_CIFAR10(self, ROOT):
        """ load all of cifar """
        xs = []
        ys = []
        for b in range(1,6):
            f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
            X, Y = self._load_CIFAR_batch(f)
            xs.append(X)
            ys.append(Y)
        Xtr = np.concatenate(xs)
        Ytr = np.concatenate(ys)
        del X, Y
        Xte, Yte = self._load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
        return Xtr, Ytr, Xte, Yte

    def reshape_images(self, images):
        images = images.reshape(len(images), 3, 32, 32)
        return images.transpose(0, 2, 3, 1)

    def get_CIFAR10_data(self, cifar10_dir, num_training=49000, num_validation=1000, num_test=10000):
        # Load the raw CIFAR-10 data
        X_train, y_train, X_test, y_test = self._load_CIFAR10(cifar10_dir)

        X_train = self.reshape_images(X_train)
        X_test = self.reshape_images(X_test)

        # Subsample the data
        mask = range(num_training, num_training + num_validation)
        X_val = X_train[mask]
        y_val = y_train[mask]
        mask = range(num_training)
        X_train = X_train[mask]
        y_train = y_train[mask]
        mask = range(num_test)
        X_test = X_test[mask]
        y_test = y_test[mask]

        x_train = X_train.astype('float32')
        x_test = X_test.astype('float32')

        x_train /= 255
        x_test /= 255

        return x_train, y_train, X_val, y_val, x_test, y_test, self.classes

    @staticmethod
    def visualize(images, labels):
        rows, columns = 5, 5
        imageId = np.random.randint(0, len(images), rows * columns)
        images = images[imageId]
        labels = [labels[i] for i in imageId]

        fig = plt.figure(figsize=(10, 10))
        for i in range(1, columns * rows + 1):
            fig.add_subplot(rows, columns, i)
            plt.imshow(images[i - 1])
            plt.xticks([])
            plt.yticks([])
            plt.title("{}".format(label_names[labels[i - 1]]))
        plt.show()


if __name__ == "__main__":
    # https://www.binarystudy.com/2021/09/how-to-load-preprocess-visualize-CIFAR-10-and-CIFAR-100.html
    # dataset source: http://www.cs.toronto.edu/~kriz/cifar.html
    # download link: http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

    parser = argparse.ArgumentParser()
    parser.add_argument('--cifar10_dir',
            help='the folder with the extracted files from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            required=True)
    args = parser.parse_args()

    cifar10_dir = args.cifar10_dir #r'C:\Users\ch3nk\Desktop\technion\deep_learning\deep_learning_046211_hw\final_project\datasets\cifar-10\cifar-10-batches-py'
    x_train, y_train, x_val, y_val, x_test, y_test, label_names = LoadCifar10().get_CIFAR10_data(cifar10_dir)

    print('Train data shape: ', x_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', x_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', x_test.shape)
    print('Test labels shape: ', y_test.shape)

    LoadCifar10.visualize(images=x_train, labels=y_train)
