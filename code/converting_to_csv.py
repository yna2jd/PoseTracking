import pandas as pd
import numpy as np
import scipy.io
from testing_mat import *
import os

def main():
    mat = scipy.io.loadmat("FLIC/examples.mat")
    dataset = mat["examples"].squeeze()
    dir = "FLIC/images"
    keypoint_names = [
        'lsho', 'lelb', 'lwri',
        'rsho', 'relb', 'rwri',
        'rhip', 'lhip',
        'leye', 'reye', 'nose',
    ]
    with open("FLIC.csv", "w") as f:
        f.write("img_name,lsho_x,lsho_y,lelb_x,lelb_y,lwri_x,lwri_y,rsho_x,rsho_y,relb_x,relb_y,rwri_x,rwri_y,rhip_x,rhip_y,lhip_x,lhip_y,leye_x,leye_y,reye_x,reye_y,nose_x,nose_y,\n")

    with open("FLIC.csv", "a") as f:
        for i in range(len(dataset)):
            cur_data = dataset[i]
            img_name = cur_data[3][0]
            s = f"{img_name},"

            coords = cur_data[2]
            joints = list(return_joints(keypoint_names, coords))

            for j in range(len(joints)):
                x, y = str(joints[j][0]), str(joints[j][1])
                s += x + "," + y + ","
            s += "\n"
            f.write(s)
    
    # print(joints)

if __name__ == "__main__":
    main()