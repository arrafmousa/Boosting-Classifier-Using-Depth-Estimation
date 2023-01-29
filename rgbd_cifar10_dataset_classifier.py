# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import os
import time
import argparse
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
from create_cifar10_depth_dataset import get_data_loaders, plot_dataloader_images


class Net(nn.Module):
    def __init__(self, add_depth):
        super().__init__()
        if add_depth:
            self.conv1 = nn.Conv2d(4, 6, 5)
        else:
            self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_cifar10(cifar10_rgbd_output_dir, add_depth, batch_size, max_train_size):
    # The output of torchvision datasets are PILImage images of range [0, 1]
    # . We transform them to Tensors of normalized range [-1, 1].
    if add_depth:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
             ])
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])

    # get data loaders
    trainloader, testloader, valloader, labels_classes = \
        get_data_loaders(cifar10_rgbd_output_dir, batch_size, max_train_size, add_depth, transform)

    return trainloader, testloader, valloader, labels_classes


def calculate_accuracy_and_loss(model, dataloader, device, loss_criterion):
    model.eval()  # put in evaluation mode
    total_correct = 0
    total_images = 0
    total_loss = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            predicted_outputs = model(images)

            # calculate accuracy
            _, predicted = torch.max(predicted_outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            # calculate loss
            total_loss += loss_criterion(predicted_outputs, labels)


    model_loss = total_loss / total_images
    # print("total_loss: ", total_loss)
    # print("model_loss: ", model_loss)
    model_accuracy = total_correct / total_images * 100
    return model_accuracy, model_loss


def train(model, train_loader, val_loader, device, epochs, models_output_dir, preview_graphs_and_images):
    # Define a Loss function and optimizer:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # training loop
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_train_accuracy = []
    epoch_val_accuracy = []
    best_val_accuracy = 0.0
    best_model = None

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_time = time.time()
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            # send them to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            outputs = model(inputs)  # forward pass
            loss = criterion(outputs, labels)  # calculate the loss

            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()  # backpropagation
            optimizer.step()  # update parameters

        # Calculate training/validation set accuracy of the existing model
        train_accuracy, train_loss = calculate_accuracy_and_loss(model, train_loader, device, criterion)
        val_accuracy, val_loss = calculate_accuracy_and_loss(model, val_loader, device, criterion)

        epoch_train_losses.append(train_loss)
        epoch_val_losses.append(val_loss)

        epoch_train_accuracy.append(train_accuracy)
        epoch_val_accuracy.append(val_accuracy)

        log = "Epoch: {} | Train loss: {:.4f} | Validation loss: {:.4f} | Training accuracy: {:.3f}% | Validation accuracy: {:.3f}% | ".format(epoch, train_loss, val_loss, train_accuracy, val_accuracy)
        epoch_time = time.time() - epoch_time
        log += "Epoch Time: {:.2f} secs".format(epoch_time)
        print(log)

        if val_accuracy > best_val_accuracy:
            best_model = model
            best_val_accuracy = val_accuracy

    print('Finished Training.')
    print("Validation set accuracy ({} images): {:.2f}%".format(len(train_loader), best_val_accuracy))

    plot_loss_and_accuracy(epoch_train_losses, epoch_val_losses, epoch_train_accuracy, epoch_val_accuracy, preview_graphs_and_images, models_output_dir)

    # save the log to file
    with open(models_output_dir / "train_log.txt", 'w') as f:
        f.write(log)

    # save the best model
    checkpoint_path = models_output_dir / 'cifar10_best_model.pth'
    torch.save(best_model.state_dict(), checkpoint_path)

    return best_model


def plot_loss_and_accuracy(epoch_train_losses, epoch_test_losses, epoch_train_accuracy, epoch_test_accuracy, preview_graphs_and_images, models_output_dir):
    plt.figure()

    # plot the validation and test loss
    plt.subplot(121)
    plt.plot(epoch_train_losses, label='train loss')
    plt.plot(epoch_test_losses, label='validation loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss VS Epochs")
    plt.xticks(range(0, len(epoch_test_losses)))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # plot the train and test accuracy
    plt.subplot(122)
    plt.plot(epoch_train_accuracy, label='train accuracy')
    plt.plot(epoch_test_accuracy, label='validation accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy VS Epochs")
    plt.xticks(range(0, len(epoch_test_accuracy)))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(models_output_dir / "loss_acc_graphs.jpg")
    plt.close()

    if preview_graphs_and_images:
        plt.show()


def evaluate_model_performance(model, device, testloader, labels_classes):
    print_predictions_for_each_class = False

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)

            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model_accuracy = float(correct) / total
    print(f'Test set accuracy ({total} images): {100 * model_accuracy} %')

    if print_predictions_for_each_class:
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in labels_classes}
        total_pred = {classname: 0 for classname in labels_classes}

        with torch.no_grad():
            for data in testloader:
                # images, labels = data
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[labels_classes[label]] += 1
                    total_pred[labels_classes[label]] += 1

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    return model_accuracy


def train_model(model, device, cifar10_rgbd_output_dir, max_dataset_size, epochs, batch_size, use_depth_images):
    preview_graphs_and_images = True

    trainloader, testloader, valloader, labels_classes = \
        load_cifar10(cifar10_rgbd_output_dir, use_depth_images, batch_size, max_dataset_size)

    if preview_graphs_and_images:
        plot_dataloader_images(trainloader, labels_classes)

    # create output dir
    results_dir = Path(os.getcwd()) / "results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp_str = time.strftime("%Y%m%d-%H%M%S")
    models_output_dir = results_dir / ("output_" + (timestamp_str) + \
                        f"_max_dataset_size={max_dataset_size}_" + "use_depth_images=" + str(use_depth_images) + f"_epochs={epochs}")
    os.makedirs(models_output_dir, exist_ok=True)

    best_model = train(model, trainloader, valloader, device, epochs, models_output_dir, preview_graphs_and_images)

    model_accuracy = evaluate_model_performance(best_model, device, testloader, labels_classes)

    return model_accuracy

    # # load back in our saved model
    # net = Net(use_depth_images)
    # net.load_state_dict(torch.load(PATH))


def plot_models_size_vs_accuracy(train_sizes, models_accuracy_rgb, models_accuracy_rgbd):
    # plt.cla()  # Clear axis
    # plt.clf()  # Clear figure
    # plt.close()

    plt.figure(3)
    plt.scatter(train_sizes, models_accuracy_rgb)
    plt.plot(train_sizes, models_accuracy_rgb, label='accuracy rgb')
    plt.scatter(train_sizes, models_accuracy_rgbd)
    plt.plot(train_sizes, models_accuracy_rgbd, label='accuracy rgbd')
    plt.xlabel("Dataset-size")
    plt.ylabel("Accuracy")
    plt.title("Accuracy VS Dataset-Size")
    plt.grid(True)
    plt.legend()
    plt.ylim([0, 1])
    plt.tight_layout()

    # plt.savefig(models_output_dir / "loss_acc_graphs.jpg")
    plt.show()


def get_model_accuracy_vs_dataset_size(cifar10_rgbd_output_dir, epochs, batch_size, dataset_sizes):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("using device: ", device)

    models_accuracy_rgb = []
    models_accuracy_rgbd = []
    for data_size in dataset_sizes:
        print("-"*20 + f"RGB - dataset size: {data_size}" + "-"*20)

        use_depth_images = False
        model = Net(use_depth_images).to(device)
        models_accuracy_rgb.append(train_model(model, device, cifar10_rgbd_output_dir, data_size, epochs, batch_size, use_depth_images))

        print("-" * 20 + f"RGBD - train size: {data_size}" + "-" * 20)
        use_depth_images = True
        model = Net(use_depth_images).to(device)
        models_accuracy_rgbd.append(train_model(model, device, cifar10_rgbd_output_dir, data_size, epochs, batch_size, use_depth_images))

    plot_models_size_vs_accuracy(dataset_sizes, models_accuracy_rgb, models_accuracy_rgbd)


def get_model_improvment_vs_dataset_size(cifar10_rgbd_output_dir, epochs, batch_size, dataset_sizes):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("using device: ", device)

    max_data_size = 60_000

    pth_paths = []

    results_dir = Path(os.getcwd()) / "results"
    results_folders = [results_dir/dir_name for dir_name in os.listdir(results_dir) if "use_depth_images" in dir_name]
    for result_folder in results_folders:
        # print(result_folder)
        pth_paths.append([result_folder / dir_name for dir_name in os.listdir(result_folder) if ".pth" in dir_name][0])

    accuracy_ratio = []
    for data_size in dataset_sizes:
        print("-"*20 + f"RGB - dataset size: {data_size}" + "-"*20)
        use_depth_images = False
        tt = f"_max_dataset_size={data_size}_" + "use_depth_images=" + str(use_depth_images) + f"_epochs={epochs}"
        pth_file = None
        for pth_path in pth_paths:
            if tt in str(pth_path):
                pth_file = pth_path
                break
        net = Net(use_depth_images).to(device)
        net.load_state_dict(torch.load(pth_file))

        trainloader, testloader, valloader, labels_classes = \
            load_cifar10(cifar10_rgbd_output_dir, use_depth_images, batch_size, max_data_size)
        rgb_model_accuracy = evaluate_model_performance(net, device, testloader, labels_classes)

        ###################

        print("-" * 20 + f"RGBD - train size: {data_size}" + "-" * 20)
        use_depth_images = True
        tt = f"_max_dataset_size={data_size}_" + "use_depth_images=" + str(use_depth_images) + f"_epochs={epochs}"
        pth_file = None
        for pth_path in pth_paths:
            if tt in str(pth_path):
                pth_file = pth_path
                break
        net = Net(use_depth_images).to(device)
        net.load_state_dict(torch.load(pth_file))

        trainloader, testloader, valloader, labels_classes = \
            load_cifar10(cifar10_rgbd_output_dir, use_depth_images, batch_size, max_data_size)
        rgbd_model_accuracy = evaluate_model_performance(net, device, testloader, labels_classes)

        accuracy_ratio.append( ((rgbd_model_accuracy/rgb_model_accuracy)-1)*100 )

    print(accuracy_ratio)

    average_improvement = sum(accuracy_ratio) / len(accuracy_ratio)

    plt.plot(dataset_sizes, accuracy_ratio)
    plt.title(f"New method improvement \n average improvement = {average_improvement:.1f}%")
    plt.xlabel("Dataset-size")
    plt.ylabel("Accuracy improvement [%]")
    plt.ylim(bottom=0)
    plt.draw()
    plt.show()


def main(args):
    epochs = 35
    batch_size = 4
    # cifar10_rgbd_dir = Path(
    #     r'C:\Users\ch3nk\Desktop\technion\deep_learning\deep_learning_046211_hw\final_project\datasets\cifar-10_rgb_format')

    cifar10_rgbd_dir = Path(args.cifar10_rgbd_dir)

    train_with_all_the_data = args.train_with_all_the_data_with_estimated_depth_images \
                              or args.train_with_all_the_data_without_estimated_depth_images
    if train_with_all_the_data:
        if args.train_with_all_the_data_with_estimated_depth_images:
            use_depth_images = True
        else:
            use_depth_images = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("using device: ", device)
        model = Net(use_depth_images).to(device)

        max_train_size = float('inf')
        train_model(model, device, cifar10_rgbd_dir, max_train_size, epochs, batch_size, use_depth_images)

    #####

    train_with_different_dataset_sizes = args.train_with_all_dataset_sizes or args.train_with_small_dataset_sizes
    if train_with_different_dataset_sizes:

        if args.train_with_all_dataset_sizes:
            dataset_sizes = list(range(2_000, 60_001, 2_000))
        else:
            dataset_sizes = list(range(500, 8_001, 500))

        get_model_accuracy_vs_dataset_size(cifar10_rgbd_dir, epochs, batch_size, dataset_sizes)

        get_model_improvment_vs_dataset_size(cifar10_rgbd_dir, epochs, batch_size, dataset_sizes)


def arguments_sanity_check(args):
    if (int(args.train_with_all_the_data_with_estimated_depth_images) +
        int(args.train_with_all_the_data_without_estimated_depth_images) +
        int(args.train_with_all_dataset_sizes) +
        int(args.train_with_small_dataset_sizes)) != 1:
        raise ValueError('choose one option: --train_with_all_the_data_with_estimated_depth_images '
                         'OR --train_with_all_the_data_without_estimated_depth_images '
                         'OR --train_with_all_dataset_sizes'
                         'OR --train_with_small_dataset_sizes')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cifar10_rgbd_dir', help='the folder created by create_cifar10_depth_dataset.py', required=True)
    parser.add_argument('--train_with_all_the_data_with_estimated_depth_images', action='store_true')
    parser.add_argument('--train_with_all_the_data_without_estimated_depth_images', action='store_true')
    parser.add_argument('--train_with_all_dataset_sizes', action='store_true', help='dataset sizes 2000 to 60000')
    parser.add_argument('--train_with_small_dataset_sizes', action='store_true', help='dataset sizes 500 to 8000')
    args = parser.parse_args()

    arguments_sanity_check(args)

    main(args)
