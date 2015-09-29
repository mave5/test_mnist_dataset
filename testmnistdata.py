__author__ = 'mra'


import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import mnist


# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples/test_mnist_data/

sys.path.insert(0, caffe_root + 'python')
import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


if not os.path.isfile(caffe_root + 'examples/mnist/lenet_iter_10000.caffemodel'):
    print("Cannot find caffemodel...")


caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'examples/mnist/lenet.prototxt',
                caffe_root + 'examples/mnist_lmdb/lenet_iter_10000.caffemodel',
                caffe.TEST)


# set net to batch size of n1
n1=100
net.blobs['data'].reshape(n1,1,28,28)


# load mnist databse
mnist_path = caffe_root+"data/mnist"
print('loading test dataset ...')
images_test,labels_test=mnist.load_mnist("testing",np.arange(10),mnist_path)

# feed data to the network
images_test=images_test.reshape(len(images_test),1,28,28)
net.blobs['data'].data[...] =images_test[0:n1,:,:]

# perfrom forward operation
out = net.forward()

print("Predictions are #{}.".format(out['prob'].argmax(axis=1)))
print("The labels  are:#{}.".format(labels_test[0:n1,0]))


def compare_listcomp(x, y):
    z=[i for i, j in zip(x, y) if i == j]
    accuracy=float(len(z))/len(x)
    return accuracy 	

x=out['prob'].argmax(axis=1)
y=labels_test[0:n1,0]
print("accuracy is:" , compare_listcomp(x,y))




