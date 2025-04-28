import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    df = pd.read_csv("FLIC.csv")
    print(df)

if __name__ == "__main__":
    main()