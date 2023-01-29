# Deep_final_project
By Mousa Arraf & Chen Chen Katz. 

This repository is part of a final project in the Technion's course 046211 - Deep Learning

## Abstract
 In this work we try to boost the accuracyof basic CNNs and CNN with an LSTM layer using an estimation of the depth of the objects in RGB pictures.
 
![alt text](https://github.com/arrafmousa/Deep_final_project/blob/main/graphics/layout.png?raw=True)
## Datasets

cifar-10 RGB format
which is available at : https://www.cs.toronto.edu/~kriz/cifar.html

Object-RGBD
which is available at : https://rgbd-dataset.cs.washington.edu/dataset.html


## Run code

1. git clone https://github.com/arrafmousa/Deep_final_project
after cloning in the repository please run :
    ```
    conda env create -f environment.yml
    conda activate depth_estimation
    ```
3. download the rgb dataset from: http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
2. run 
    ```
    # create cifar-10 RGBD dataset:
    python create_cifar10_depth_dataset.py --cifar10_dir [the rgb dataset you downloaded] --cifar10_RGBD_output_dir [rgbd output dir]
    
    # get the accuracy improvment ratio of diffrent datasets sizes in case of using estimated depth images:
    python rgbd_cifar10_dataset_classifier.py --cifar10_rgbd_dir  [the created rgbd dataset path] --train_with_all_dataset_sizes
    
    # get the accuracy improvment ration of diffrent small datasets sizes in case of using estimated depth images: 
    python rgbd_cifar10_dataset_classifier.py --cifar10_rgbd_dir  [the created rgbd dataset path] --train_with_small_dataset_sizes

    # test the performance of the diffrent architectures
    # where you can pass True or False as is_lstm or is_depth to train diffrent models.
    python conv_neural_network.py --is_lstm=is_depth --is_depth=is_depth --cifar_dir=[the created rgbd dataset path]
    ```

## Results 
here we can see that using estimated depth we were able to boost the performance of our baseline models.

![alt text](https://github.com/arrafmousa/Deep_final_project/blob/main/graphics/Screenshot%202023-01-23%20213336.png?raw=True)

 we also experimented with diffrent sizes of datasets the accuracy boost is as follows
 ![alt text](https://github.com/arrafmousa/Deep_final_project/blob/main/graphics/Picture1.png?raw=True)

 As expected, the biggest improvement in performance can be achieved when using small datasets
 ![alt text]( https://github.com/arrafmousa/Deep_final_project/blob/main/graphics/Picture2.png?raw=True)


