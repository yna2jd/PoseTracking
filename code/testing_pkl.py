import pickle
import scipy.io 
import random
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

def main():
    rot = "numtimesrot90clockwise"
    cal = "calibration"
    with open("sportspose/calib.pkl", "rb") as f:
        data = pickle.load(f)
    data_cal = data[cal]


    # for i in range(len(data_cal)):
    #     print(f"{i}: {type(data_cal)} -- {data_cal[i]}")
    for i in data_cal:
        for j in i.keys():
            print(len(i[j]), end=", ")
        print()

if __name__ == "__main__":
    main()