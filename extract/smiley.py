from glob import glob
import numpy as np
import cv2

def tanh(x):
	t = (np.exp(x)- np.exp(-x))/(np.exp(x) + np.exp(-x))
	return t

def dtanh(x):
	dt = 1 - (tanh(x)**2)
	return dt

def sigmoid(x):
	s = 1/(1+np.exp(-x))
	return s

def dsigmoid(x):
	ds = sigmoid(x)
	return ds*(1-ds)

def load_images():
	accumulator = []
	for i in range(2):
		im_files = glob('./data/{}/*.png'.format(i))
		img = cv2.imread(im_files[0], 0) # 0 for B&W
		img = img.flatten()
		img = np.append(img, [i], axis=0)
		images = np.array([img])
		for j in range(1, len(im_files)):
			img = cv2.imread(im_files[j], 0) # (1, 784) B&W
			img = img.flatten() #reshape(1, img.shape[0]*img.shape[1])[0] # turning into a vector
			img = np.append(img, [i], axis=0)
			images = np.append(images, [img], axis=0) # (m, 2352)
			# STRAIGHTEN IMAGE TO A NUMPY ARRAY
			# ADD TO A GLOBAL ARRAY
			# y = images[i][len(images[0])-1]
			# After adding both options
		accumulator.insert(i, images)
	data = np.append(accumulator[0], accumulator[1], axis=0)
	np.random.shuffle(data)
	X = data[:, :len(data[0])-1].T
	Y = data[:, len(data[0])-1].reshape(1, len(data))
	return (X, Y)

## LOAD X AND Y
X, Y = load_images()
n_x = len(X)
n_h = int(2*n_x)
n_y = len(Y)

## INITIALIZE PARAMETERS W AND b ##
w1 = np.random.rand(n_h, n_x)*0.01
b1 = np.zeros((n_h, 1))
w2 = np.random.rand(n_y, n_h)*0.01
b2 = np.zeros((n_y, 1))

## ITERATE UNTIL CONVERGENCE
for i in range(10000):
	print("Iteration : ", i)
	## FORWARD PROPAGATION ##
	z1 = np.dot(w1, X) + b1
	a1 = sigmoid(z1)
	z2 = np.dot(w2, a1) + b2
	a2 = sigmoid(z2)

	## BACKPROPAGATION ##
	dz2 = a2 - Y
	dw2 = np.dot(dz2, a1.T)
	db2 = dz2
	dz1 = np.dot(w2.T, dz2)*dsigmoid(z1)
	dw1 = np.dot(dz1, X.T)
	db1 = dz1

	## UPDATE PARAMETERS ##
	learning_rate = 0.01
	w1 = w1 - learning_rate*dw1
	b1 = b1 - learning_rate*db1
	w2 = w2 - learning_rate*dw2
	b2 = b2 - learning_rate*db2

## WRITE THE PARAMETERS TO A FILE ##

## TESTING WITH AN OVERFITTED MODEL ##
x = np.array([X[11:12]])
print(x)
z1 = np.dot(w1, x) + b1
a1 = sigmoid(z1)
z2 = np.dot(w2, a1) + b2
a2 = sigmoid(z2)
print("Y : ", Y[11:12])
print("a2 : ", a2[11:12])