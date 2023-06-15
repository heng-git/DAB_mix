import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from util.utils import renorm
from util.misc import color_sys
import os.path
from typing import Any, Callable, Optional, Tuple, List
import datasets.transforms as T
from torchvision import transforms
from PIL import Image

#from .vision import VisionDataset

_color_getter = color_sys(100)
CLASSES = ['N/A', 'Person', 'Bicycle', 'Car', 'Motorcycle', 'Airplane', 'Bus', 'Train', 'Truck', 'Boat', 'Traffic-Light', 'Fire-Hydrant', 'N/A', 'Stop-Sign', 'Parking Meter', 'Bench', 'Bird', 'Cat', 'Dog', 'Horse', 'Sheep', 'Cow', 'Elephant', 'Bear', 'Zebra', 'Giraffe', 'N/A', 'Backpack', 'Umbrella', 'N/A', 'N/A', 'Handbag', 'Tie', 'Suitcase', 'Frisbee', 'Skis', 'Snowboard', 'Sports-Ball', 'Kite', 'Baseball Bat', 'Baseball Glove', 'Skateboard', 'Surfboard', 'Tennis Racket', 'Bottle', 'N/A', 'Wine Glass', 'Cup', 'Fork', 'Knife', 'Spoon', 'Bowl', 'Banana', 'Apple', 'Sandwich', 'Orange', 'Broccoli', 'Carrot', 'Hot-Dog', 'Pizza', 'Donut', 'Cake', 'Chair', 'Couch', 'Potted Plant', 'Bed', 'N/A', 'Dining Table', 'N/A','N/A', 'Toilet', 'N/A', 'TV', 'Laptop', 'Mouse', 'Remote', 'Keyboard', 'Cell-Phone', 'Microwave', 'Oven', 'Toaster', 'Sink', 'Refrigerator', 'N/A', 'Book', 'Clock', 'Vase', 'Scissors', 'Teddy-Bear', 'Hair-Dryer', 'Toothbrush']
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933]
]
# plot known and unknown box
def add_box_to_img(img, boxes, colorlist, brands=None):
    """[summary]

    Args:
        img ([type]): np.array, H,W,3
        boxes ([type]): list of list(4)
        colorlist: list of colors.
        brands: text.

    Return:
        img: np.array. H,W,3.
    """
    H, W = img.shape[:2]
    for _i, (box, color) in enumerate(zip(boxes, colorlist)):
        x, y, w, h = box[0] * W, box[1] * H, box[2] * W, box[3] * H
        img = cv2.rectangle(img.copy(), (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), color, 2)
        if brands is not None:
            brand = brands[_i]
            org = (int(x-w/2), int(y+h/2))
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            thickness = 1
            img = cv2.putText(img.copy(), str(brand), org, font, 
                fontScale, color, thickness, cv2.LINE_AA)
    return img

def plot_dual_img(img, boxes, labels, idxs, probs=None):
    """[summary]

    Args:
        img ([type]): 3,H,W. tensor.
        boxes (): tensor(Kx4) or list of tensor(1x4).
        labels ([type]): list of ints.
        idxs ([type]): list of ints.
        probs (optional): listof floats.

    Returns:
        img_classcolor: np.array. H,W,3. img with class-wise label.
        img_seqcolor: np.array. H,W,3. img with seq-wise label.
    """
    # import ipdb; ipdb.set_trace()
    boxes = [i.cpu().tolist() for i in boxes]
    img = (renorm(img.cpu()).permute(1,2,0).numpy() * 255).astype(np.uint8)
    # plot with class
    class_colors = [_color_getter(i) for i in labels]
    if probs is not None:
        brands = ["{},{:.2f}".format(j,k) for j,k in zip(labels, probs)]
    else:
        brands = labels
    img_classcolor = add_box_to_img(img, boxes, class_colors, brands=brands)
    # plot with seq
    seq_colors = [_color_getter((i * 11) % 100) for i in idxs]
    img_seqcolor = add_box_to_img(img, boxes, seq_colors, brands=idxs)
    return img_classcolor, img_seqcolor


def plot_raw_img(img, boxes, labels):
    """[summary]

    Args:
        img ([type]): 3,H,W. tensor. 
        boxes ([type]): Kx4. tensor
        labels ([type]): K. tensor.

    return:
        img: np.array. H,W,3. img with bbox annos.
    
    """
    img = (renorm(img).permute(1,2,0).numpy() * 255).astype(np.uint8)
    H, W = img.shape[:2]
    for box, label in zip(boxes.tolist(), labels.tolist()):
        x, y, w, h = box[0] * W, box[1] * H, box[2] * W, box[3] * H
        # import ipdb; ipdb.set_trace()
        img = cv2.rectangle(img.copy(), (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), _color_getter(label), 2)
        # add text
        org = (int(x-w/2), int(y+h/2))
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 1
        plt.axis('off')  # 关掉坐标轴为 off
        img = cv2.putText(img.copy(), str(label), org, font, 
            fontScale, _color_getter(label), thickness, cv2.LINE_AA)

    return img


def plot_results(pil_img, prob, boxes, labels=True):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()

    for prob, (x0, y0, x1, y1), color in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,fill=False, color=color, linewidth=2))
        cl = prob.argmax()
        text = f'{CLASSES[cl]}: {prob[cl]:0.2f}'
        if labels:
            ax.text(x0, y0, text, fontsize=15,bbox=dict(facecolor=color, alpha=0.75))
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    img = Image.open(r'/Object_detection/ZSH/datasets/COCOmini/mini_train2017/000000001146.jpg')
    #img_arr = np.array(img)
    box = torch.tensor([[200,200,2000,2000]])
    color = torch.tensor([[0.2,0.2,0.2,1.0]])
    label = torch.tensor([1])
    index = torch.tensor([0])
    probs = torch.tensor([[0,0.1,0.5,0.4,0.0,0.0,0.0]])
    handle= plot_results(img,probs,box)
    # plt.axis('off')
    # plt.imshow(img)
    # plt.show()
    # cv2.namedWindow("Demo3", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Demo3", 400, 300)
    # cv2.imshow("Demo3", img)
    # cv2.waitKey()