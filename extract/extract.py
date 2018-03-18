# install opencv: https://www.solarianprogrammer.com/2016/09/17/install-opencv-3-with-python-3-on-windows/
from glob import glob
import numpy as np
import cv2

def print_image():
	img = cv2.imread('image.png', 0)
	print(img.shape)
	for i in range(img.shape[0]):	
		for j in range(img.shape[1]):
			if img[i][j]>100:
				print('_', end=" ")
			else: print('#', end=" ")
		print()
	#return (X, Y)

def load_images():
	accumulator = []
	for i in range(2):
		im_files = glob('./data/{}/*.png'.format(i))
		img = cv2.imread(im_files[0], 0) # 0 for B&W
		img = img.flatten()
		img = np.append(img, [i], axis=0)
		images = np.array([img])
		#print(images.shape)
		#print(im_files[0])
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
	""" 
	# SPLITTING THE DATASET IN TRAIN AND TEST
	r = 1.0
	k = int(len(data)*r)
	#print("k: ",k)
	train_data = data[:k,:]
	test_data = data[k:, :]
	print("train : ", len(train_data), " test : ", len(test_data))
	X_train = train_data[:, :len(train_data[0])-1].T
	Y_train = train_data[:, len(train_data[0])-1].reshape(1, len(train_data))
	X_test = test_data[:, :len(test_data[0])-1].T
	Y_test = test_data[:, len(test_data[0])-1].reshape(1, len(test_data))"""
	X = data[:, :len(data[0])-1].T
	Y = data[:, len(data[0])-1].reshape(1, len(data))
	#return (X_train, Y_train, X_test, Y_test)# THE GLOBAL ARRAY
	return (X, Y)


#print_image()
#X_train, Y_train,  X_test, Y_test = load_images()
X, Y = load_images()
print(X.shape)
print(Y.shape)
print(Y)
