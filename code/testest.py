import cv2

path = "FLIC/images/2-fast-2-furious-00019871.jpg"
img = cv2.imread(path)

print(path)
cv2.imshow(path, img)
cv2.waitKey(0)