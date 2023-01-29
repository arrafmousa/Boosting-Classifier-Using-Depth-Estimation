import cv2
import torch
import os
import matplotlib.pyplot as plt
from pathlib import Path
import random


def get_all_images_paths_in_dir(dir_path, image_type=".png"):
    img_files = os.listdir(dir_path)
    if image_type is not None:
        img_files = list(filter(lambda x: image_type in x, img_files))
    img_files.sort(key=lambda x: int(x.split(".png")[0]))
    return [Path(dir_path) / Path(file_name) for file_name in img_files]


class ImagToPlot:
    def __init__(self, image_name, original_image, depth_image):
        self.image_name = image_name
        self.original_image = original_image
        self.depth_image = depth_image


def plot_images(images_to_plot):
    num_of_images = len(images_to_plot)
    if num_of_images > 0:
        plt.figure()
        for ind, image_to_plot in enumerate(images_to_plot):
            plt.subplot(num_of_images, 2, 2 * ind + 1)
            plt.title(image_to_plot.image_name)
            plt.imshow(image_to_plot.original_image)
            plt.xticks([])
            plt.yticks([])

            plt.subplot(num_of_images, 2, 2 * ind + 2)
            plt.title(f"{image_to_plot.image_name} estimated depth image")
            plt.imshow(image_to_plot.depth_image)#, vmin=image_to_plot.depth_image.flatten().min(), vmax=image_to_plot.depth_image.flatten().max())
            plt.xticks([])
            plt.yticks([])

        plt.suptitle('Images and their estimated depth images', fontsize=12)
        plt.show()


def save_image(output_path, image):
    print("saving image: ", str(output_path))
    plt.imsave(output_path, image, cmap='gray')


def create_depth_images(input_dir, output_dir, model_type="DPT_BEiT_L_512", num_of_images_to_plot=0):
    files_paths = get_all_images_paths_in_dir(input_dir)

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("using device: ", device)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    elif model_type == "DPT_BEiT_L_512":
        transform = midas_transforms.beit512_transform
    else:
        transform = midas_transforms.small_transform

    indices_to_plot = random.sample(range(0, len(files_paths)), num_of_images_to_plot)
    images_to_plot = []

    for ind, path in enumerate(files_paths):
        depth_output_path = str(Path(output_dir) / path.name)
        if os.path.isfile(depth_output_path):
            continue

        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

        input_batch = transform(img).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            prediction = prediction.squeeze()

        output = prediction.cpu().numpy()

        # normalize
        output = (output-output.flatten().min())/(output.flatten().max() - output.flatten().min())

        save_image(output_path=depth_output_path, image=output)

        if ind in indices_to_plot:
            images_to_plot.append(ImagToPlot(path.name, img, output))

    plot_images(images_to_plot)


def create_dataset(input_dir):
    model_type = "DPT_BEiT_L_512"  # MiDaS 3.1 highest quality
    # model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    print_available_models = False
    if print_available_models:
        models = torch.hub.list("intel-isl/MiDaS")
        print("available models: ", models)

    create_depth_images(input_dir=str(input_dir),
                        output_dir=str(input_dir) + "_depth",
                        model_type=model_type,
                        num_of_images_to_plot=0)


if __name__ == "__main__":
    input_dir = Path.cwd() / "dataset"
    create_dataset(input_dir)
