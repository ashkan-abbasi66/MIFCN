import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import cv2

# **********************************
# ********** SETTINGS **************
# **********************************

hp=400 # This constant is used for the weighted averaging module (Default: 400)

N_train=10 # number of training images
N_patches=400   # number of patches that was extracted from each image
N_blks=5 # number of similar patches for each patch
N_test=18   # test dataset size

N_epochs=60
N_FM=24 # Number of feature maps
k_size=3    # kernel size for convolution layers

# **********************************

def lrelu(x):
    return tf.maximum(x*0.2,x)

def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0]//2, shape[1]//2
        for i in range(shape[2]):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer

def build(input):
    # input has 5 channels
    i1,i2,i3,i4,i5=tf.split(input,5,axis=3)

    # branch 1 (Main branch)
    net1 = slim.conv2d(i1, N_FM, [k_size, k_size], rate=1,
                       activation_fn=lrelu,
                       weights_initializer=identity_initializer(),
                       scope='net1_1')
    net1 = slim.conv2d(net1, N_FM, [k_size, k_size], rate=2,
                       activation_fn=lrelu,
                       weights_initializer=identity_initializer(),
                       scope='net1_2')
    net1 = slim.conv2d(net1, N_FM, [k_size, k_size], rate=1,
                       activation_fn=lrelu,
                       weights_initializer=identity_initializer(),
                       scope='net1_3')
    net1 = slim.conv2d(net1, 1, [1, 1], rate=1,
                       activation_fn=None,
                       scope='net1_last')

    # branch 2
    net2 = slim.conv2d(i2, N_FM, [k_size, k_size], rate=1,
                       activation_fn=lrelu,
                       weights_initializer=identity_initializer(),
                       scope='net2_1')
    net2 = slim.conv2d(net2, N_FM, [k_size, k_size], rate=2,
                       activation_fn=lrelu,
                       weights_initializer=identity_initializer(),
                       scope='net2_2')
    net2 = slim.conv2d(net2, N_FM, [k_size, k_size], rate=1,
                       activation_fn=lrelu,
                       weights_initializer=identity_initializer(),
                       scope='net2_3')
    net2 = slim.conv2d(net2, 1, [1, 1], rate=1,
                       activation_fn=None,
                       scope='net2_last')

    # branch 3
    net3 = slim.conv2d(i3, N_FM, [k_size, k_size], rate=1,
                       activation_fn=lrelu,
                       weights_initializer=identity_initializer(),
                       scope='net3_1')
    net3 = slim.conv2d(net3, N_FM, [k_size, k_size], rate=2,
                       activation_fn=lrelu,
                       weights_initializer=identity_initializer(),
                       scope='net3_2')
    net3 = slim.conv2d(net3, N_FM, [k_size, k_size], rate=1,
                       activation_fn=lrelu,
                       weights_initializer=identity_initializer(),
                       scope='net3_3')
    net3 = slim.conv2d(net3, 1, [1, 1], rate=1,
                       activation_fn=None,
                       scope='net3_last')

    # branch 4
    net4 = slim.conv2d(i4, N_FM, [k_size, k_size], rate=1,
                       activation_fn=lrelu,
                       weights_initializer=identity_initializer(),
                       scope='net4_1')
    net4 = slim.conv2d(net4, N_FM, [k_size, k_size], rate=2,
                       activation_fn=lrelu,
                       weights_initializer=identity_initializer(),
                       scope='net4_2')
    net4 = slim.conv2d(net4, N_FM, [k_size, k_size], rate=1,
                       activation_fn=lrelu,
                       weights_initializer=identity_initializer(),
                       scope='net4_3')
    net4 = slim.conv2d(net4, 1, [1, 1], rate=1,
                       activation_fn=None,
                       scope='net4_last')

    # branch 5
    net5 = slim.conv2d(i5, N_FM, [k_size, k_size], rate=1,
                       activation_fn=lrelu,
                       weights_initializer=identity_initializer(),
                       scope='net5_1')
    net5 = slim.conv2d(net5, N_FM, [k_size, k_size], rate=2,
                       activation_fn=lrelu,
                       weights_initializer=identity_initializer(),
                       scope='net5_2')
    net5 = slim.conv2d(net5, N_FM, [k_size, k_size], rate=1,
                       activation_fn=lrelu,
                       weights_initializer=identity_initializer(),
                       scope='net5_3')
    net5 = slim.conv2d(net5, 1, [1, 1], rate=1,
                       activation_fn=None,
                       scope='net5_last')

    # total=tf.concat([net1,net2,net3,net4,net5],axis=3)

    a = tf.minimum(tf.maximum(net1, 0.0), 1.0) * 255.0  # @@
    b = tf.minimum(tf.maximum(net2, 0.0), 1.0) * 255.0
    c = tf.minimum(tf.maximum(net3, 0.0), 1.0) * 255.0
    d = tf.minimum(tf.maximum(net4, 0.0), 1.0) * 255.0
    e = tf.minimum(tf.maximum(net5, 0.0), 1.0) * 255.0
    # --------------------

    d1 = (a - a) ** 2
    d2 = (a - b) ** 2
    d3 = (a - c) ** 2
    d4 = (a - d) ** 2
    d5 = (a - e) ** 2

    d1 = tf.exp(-d1 / hp)
    d2 = tf.exp(-d2 / hp)
    d3 = tf.exp(-d3 / hp)
    d4 = tf.exp(-d4 / hp)
    d5 = tf.exp(-d5 / hp)

    wsum = d1 + d2 + d3 + d4 + d5

    w1 = tf.divide(d1, wsum)
    w2 = tf.divide(d2, wsum)
    w3 = tf.divide(d3, wsum)
    w4 = tf.divide(d4, wsum)
    w5 = tf.divide(d5, wsum)

    network_total = tf.multiply(w1, a) + tf.multiply(w2, b) + tf.multiply(w3, c) + tf.multiply(w4, d) + tf.multiply(w5,
                                                                                                                    e)
    total = network_total / 255.0  # @@

    net6 = slim.conv2d(total, 24, [k_size, k_size], rate=1,
                       activation_fn=lrelu,
                       weights_initializer=identity_initializer(),
                       # weights_regularizer=slim.l1_regularizer(0.07),
                       scope='nett_1')
    net6 = slim.conv2d(net6, 1, [1, 1], rate=1,
                       activation_fn=None,
                       scope='nett_last')

    return tf.concat([net1,net2,net3,net4,net5,net6],axis=3)

def prepare_data():
    # test set
    val_inp_fnames = []
    val_lbl_fnames = []

    for dirname in ['test/Synthetic']:
        for i in range(1,N_test+1):
            val_inp_fnames.append("./data/%s/%d/test.tif"%(dirname,i))
            val_inp_fnames.append("./data/%s/%d/1.tif" % (dirname, i))
            val_inp_fnames.append("./data/%s/%d/2.tif" % (dirname, i))
            val_inp_fnames.append("./data/%s/%d/3.tif" % (dirname, i))
            val_inp_fnames.append("./data/%s/%d/4.tif" % (dirname, i))

            val_lbl_fnames.append("./data/%s/%d/Average.tif" % (dirname, i))
    return val_inp_fnames, val_lbl_fnames

def save_data():
    # USAGE:
    #     # For saving:
    #     from model_mifcn_helpers import save_data
    #     save_data()
    #
    #     # For loading:
    #     import numpy as np
    #     inputs_ndarray = np.load('data/train15_inputs.npy')
    #     labels_ndarray = np.load('data/train15_labels.npy')

    # training set
    input_fnames = []
    # label_fnames = []
    for dirname in ['train15']:
        for i in range(1, N_train + 1):  # training images
            for jj in range(1, N_patches + 1):  # patches
                for kk in range(1, N_blks + 1):  # similar patches for each patch
                    input_fnames.append("./data/%s/L%0.2d_%0.3d_%0.2d.tif" % (dirname, i, jj, kk))
                    # label_fnames.append("./data/%s/H%0.2d_%0.3d_%0.2d.tif" % (dirname, i, jj, kk))

    img_indices=range(0,len(input_fnames),N_blks)

    N_images = len(img_indices)

    inputs_ndarray = np.empty(shape=[N_train*N_patches,N_blks,15,15], dtype=np.float32)
    labels_ndarray = np.empty(shape=[N_train*N_patches,N_blks,15,15], dtype=np.float32)

    dirname = 'train15'
    for i in range(1, N_train + 1):
        print('Saving patches of the training image number %d ...'%i)
        for jj in range(1, N_patches + 1):
            for kk in range(1, N_blks + 1):
                # Low-SNR Images (noisy)
                file = "./data/%s/L%0.2d_%0.3d_%0.2d.tif" % (dirname, i, jj, kk)
                tmp = np.float32(cv2.imread(file, cv2.IMREAD_GRAYSCALE)) / 255.0
                index = (i-1)*N_patches+(jj-1)
                inputs_ndarray[index,kk-1] = tmp

                # The Corresponding High-SNR Images (clean)
                file = "./data/%s/H%0.2d_%0.3d_%0.2d.tif" % (dirname, i, jj, kk)
                tmp = np.float32(cv2.imread(file, cv2.IMREAD_GRAYSCALE)) / 255.0
                labels_ndarray[index,kk-1] = tmp
    print(inputs_ndarray.shape)
    print(labels_ndarray.shape)
    np.save('data/train15_inputs.npy',inputs_ndarray)
    np.save('data/train15_labels.npy', labels_ndarray)