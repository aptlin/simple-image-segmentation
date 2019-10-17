import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import Segmentation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        help="The image to segment relative to the current folder.",
        default="./data/test.jpg",
    )
    args = parser.parse_args()
    img = plt.imread(args.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255

    model = Segmentation(2)
    centers, labels = model.segment(gray)

    mask = (labels.reshape(*gray.shape) * 255).astype(np.uint8)

    # remove noise
    kernel = np.ones((4, 4), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)

    cv2.imshow("Segmentation map", mask)
    cv2.waitKey(0)
