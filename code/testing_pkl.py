import pickle

with open("sportspose/calib.pkl", "rb") as f:
    data = pickle.load(f)

print(data["calibration"])

