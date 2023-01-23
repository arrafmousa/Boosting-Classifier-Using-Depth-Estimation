# Deep_final_project
By Mousa Arraf & Chen Chen Katz.

This repository is part of a final project in the Technion's course 046211 - Deep Learning

![alt text](https://github.com/arrafmousa/Deep_final_project/blob/main/graphics/layout.png?raw=True)
# Datasets

cifar-10 RGB format
which is available at : https://www.cs.toronto.edu/~kriz/cifar.html
Object-RGBD
which is available at : https://rgbd-dataset.cs.washington.edu/dataset.html


## Run code
1. git clone https://github.com/arrafmousa/Deep_final_project
# after cloning in the repository please run :
    ```
    conda env create -f environment.yml
    conda activate depth_estimation
    ```
2. run 
    ```
    python conv_neural_network.py --is_lstm=is_depth --is_depth=is_depth --cifar_dir=cifar_dir
    ```
where you can pass True or False as is_lstm or is_depth to train diffrent models.
cifar_dir should be the path to the root directory of the cifar dataset
