import scipy.io

mat = scipy.io.loadmat("FLIC/examples.mat")
print(mat)

mat = [ [i for i in X] for X in mat["obj_contour"] ]