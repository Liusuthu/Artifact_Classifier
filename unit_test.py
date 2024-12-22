import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def test_data_loader():
    from torchvision.utils import make_grid

    from datasets import get_data_loader

    train_loader = get_data_loader("data", "train", batch_size=8, num_workers=0, augment=False)
    num_classes = len(train_loader.dataset.classes)
    assert num_classes == 4, f"Expected 4 classes, got {num_classes}."

    images, labels = next(iter(train_loader))
    # print labels
    print(" ".join(chr(65 + x) for x in labels))
    # show images
    imshow(make_grid(images))


def test_stn():
    from networks import STN

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stn = STN(3).to(device).eval()
    data_in = torch.randn(1, 3, 32, 32).to(device)
    with torch.no_grad():
        data_out = stn(data_in)
        data_diff = torch.abs(data_in - data_out)
        #print("data_diff=",data_diff)
    assert torch.all(
        data_diff < 1e-6
    ), "STN forward check failed. Please check the network implementation and weight initialization."
    print("STN forward check passed.")


def imshow(img):
    img = img / 2 + 0.5  # denormalize
    npimg = img.numpy()
    plt.figure(figsize=(8, 2))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    os.makedirs("visualized", exist_ok=True)
    plt.savefig("visualized/augmentation.jpg", dpi=300)
    plt.show()


def main(unit):
    if unit == "data_loader":
        test_data_loader()
    elif unit == "stn":
        test_stn()
    else:
        raise ValueError(f"Invalid unit: {unit}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("unit", type=str, choices=["data_loader", "stn"])
    args = parser.parse_args()

    main(args.unit)
