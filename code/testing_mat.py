import scipy.io 
import random
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_key():
    return {
        'lsho': 1, 'lelb': 2, 'lwri': 3, 'rsho': 4, 'relb': 5, 'rwri': 6,
        'lhip': 7, 'lkne': 8, 'lank': 9, 'rhip': 10, 'rkne': 11, 'rank': 12,
        'leye': 13, 'reye': 14, 'lear': 15, 'rear': 16, 'nose': 17,
        'msho': 18, 'mhip': 19, 'mear': 20, 'mtorso': 21, 'mluarm': 22,
        'mruarm': 23, 'mllarm': 24, 'mrlarm': 25, 'mluleg': 26, 'mruleg': 27,
        'mllleg': 28, 'mrlleg': 29,
        'L_Shoulder': 1, 'L_Elbow': 2, 'L_Wrist': 3, 'R_Shoulder': 4,
        'R_Elbow': 5, 'R_Wrist': 6, 'L_Hip': 7, 'L_Knee': 8, 'L_Ankle': 9,
        'R_Hip': 10, 'R_Knee': 11, 'R_Ankle': 12,
        'L_Eye': 13, 'R_Eye': 14, 'L_Ear': 15, 'R_Ear': 16, 'Nose': 17
    }

def lookup_part(names):
    key = get_key()
    return [key[name] - 1 for name in names]

def plot_joints(names, style, coords):
    indices = lookup_part(names)
    points = coords[:, indices]
    valid = ~np.isnan(points[0]) & ~np.isnan(points[1])
    plt.plot(points[0, valid], points[1, valid], style, linewidth=3)

def main():
    mat = scipy.io.loadmat("FLIC/examples.mat")
    examples = mat["examples"].squeeze()
    i = random.randint(0, len(examples)-1)
    example = examples[i]

    imgdir = "FLIC/images/"
    img_name = example[3][0]
    img = cv2.imread(os.path.join(imgdir, img_name))
    plt.imshow(img)
    plt.axis("off")
    # cv2.imshow("", img)
    # cv2.waitKey(0)
    
    # for i in range(len(example)):
    #     print(f"{i}: {type(example)} -- {example[i]}")

    coords = example[2]
    x1, y1, x2, y2 = example[6][0]
    # plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "w--") # torsobox

    plot_joints(['lsho', 'lelb', 'lwri'], 'go-', coords)
    plot_joints(['rsho', 'relb', 'rwri'], 'mo-', coords)
    plot_joints(['rhip', 'lhip'], 'bo-', coords)
    plot_joints(['leye', 'reye', 'nose', 'leye'], 'c.-', coords)

    plt.show()

if __name__ == "__main__":
    main()