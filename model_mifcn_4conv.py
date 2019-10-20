from __future__ import division

import os
import time

from model_mifcn_helpers import *

# This program has 4 main parts:
#   1- Settings: Note that some settings are at the beginning of this file
#       and others are at the beginning of the `model_mifcn_helpers` module.
#   2- MAIN: In the MAIN part, we construct the computation graph which
#       can be used for training and testing the method.
#   3- TRAINING: if you set `is_training` to True, the training part will run.
#   4- TEST: if you set `is_training` to False, the last model will be
#       loaded. Then, the program reconstructs the test dataset images.
# The code was heavily commented. Please see the following codes and comments.

# **********************************
# ********** SETTINGS **************
# **********************************

np.random.seed(1024)
tf.set_random_seed(1024)

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
session_conf = tf.ConfigProto(
    device_count={'CPU': 1, 'GPU': 0},
    allow_soft_placement=True,
    log_device_placement=False,
    gpu_options=gpu_options)
# session_conf = None

# If `is_training` is set to `True`, the model is trained on the training
# dataset, and the trained model after each epoch is saved in a separate
# folder which is stored in the `out_folder`.
# Also, the test images will reconstructed after the epoch numbers which
# stored in `see_valid_imgs`.
# the statistics (including epoch number, MSE loss, etc) are saved during
# training in the `scores_training.txt` and `scores_test.txt` files.
# If `is_training` is set to `False`, the latest model from the folder
# which is determined by `out_folder` will be loaded. Then, the test dataset
# images will be reconstructed and put into a seperate folder named
# `FINAL_RESULTS`.

is_training=False

out_folder="results/model_mifcn_4conv"

# Reconstruct some test images at these epochs:
see_valid_imgs = [50,60]

# **********************************
# ************ MAIN ****************
# **********************************

if os.path.isdir("%s"%(out_folder))==False:
    os.makedirs("%s"%(out_folder))

if session_conf is not None:
    sess = tf.Session(config=session_conf)
else:
    sess = tf.Session()

# Preparing the test image paths which are used
# for validation/test purposes.
val_inp_fnames,val_lbl_fnames=prepare_data()

# Placeholders (ph...)
phInput=tf.placeholder(tf.float32,shape=[None,None,None,5])
phLabel=tf.placeholder(tf.float32,shape=[None,None,None,5])
t1, t2, t3, t4, t5 = tf.split(phLabel, 5, axis=3)

# Constructing the model
net_out=build(phInput)
a, b, c, d, e,fff = tf.split(net_out, 6, axis=3)

# Defining loss functions
# This is used for training.
loss=tf.reduce_mean(tf.square(a-t1)+
                    tf.square(b-t2)+
                    tf.square(c-t3)+
                    tf.square(d-t4)+
                    tf.square(e-t5))+tf.reduce_mean(tf.square(fff-a))

# Mean square error is just used for evaluation
mse_loss=tf.reduce_mean(tf.square(fff-t1))

# optimizer
phEta=tf.placeholder(tf.float32,shape=[])
opt=tf.train.AdamOptimizer(learning_rate=phEta).\
    minimize(loss,var_list=[var for var in tf.trainable_variables()])

# Saver object is used to save the trained model
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

# How many parameters does the model have?
total_parameters = np.sum([np.prod(v.get_shape().as_list())
                           for v in tf.trainable_variables()])
print('INFO: Number of trainable variables: %d'%total_parameters)

# Restore the latest model or train from the beginning.
if tf.train.get_checkpoint_state(out_folder):
    ckpt = tf.train.latest_checkpoint(out_folder)
    saver.restore(sess,ckpt)
    print('INFO: loaded ' + ckpt)
else:
    print('INFO: Training starts from the beginning ...')
    # create a log file to save some statistics during training
    target = open("%s/scores_training.txt" % out_folder, 'a')
    msg = 'Epoch\t MSE\t MeanLoss\t Time\n'
    target.write(msg)
    target.close()
    # create another log file to save some statistics related
    # to the test dataset.
    target = open("%s/scores_test.txt" % out_folder, 'a')
    msg = 'Epoch\t MSE\t Time\n'
    target.write(msg)
    target.close()

# **********************************
# ********** TRAINING **************
# **********************************

if is_training:
    # Load the training data
    inputs_ndarray = np.load('data/train15_inputs.npy')
    labels_ndarray = np.load('data/train15_labels.npy')

    Training_Loss = 0
    MSE_Training_Loss = 0

    for epoch in range(1, N_epochs+1):
        if os.path.isdir("%s/%04d"%(out_folder,epoch)):
            continue

        start_time = time.time()

        N_samples = N_train * N_patches

        sid = range(0, N_samples*N_blks , N_blks)
        for id in np.random.permutation(sid):
            index = id //N_blks

            # Preparing a batch of training data==========================================
            N_batch = 4
            h, w = inputs_ndarray[index, 0].shape
            input_tensor = np.zeros((N_batch, h, w, N_blks))  # main slice and its nearby slices
            label_tensor = np.zeros((N_batch, h, w, N_blks))

            for slice in range(0,N_blks):
                input_tensor[0, :, :, slice] = inputs_ndarray[index, slice]
                label_tensor[0, :, :, slice] = labels_ndarray[index, slice]

            # Data Augmentation
            for slice in range(0,N_blks):
                input_tensor[1, :, :, slice] = np.fliplr(inputs_ndarray[index, slice])
                label_tensor[1, :, :, slice] = np.fliplr(labels_ndarray[index, slice])

            for slice in range(0,N_blks):
                input_tensor[2, :, :, slice] = np.flipud(inputs_ndarray[index, slice])
                label_tensor[2, :, :, slice] = np.flipud(labels_ndarray[index, slice])

            for slice in range(0,N_blks):
                input_tensor[3, :, :, slice] = np.rot90(inputs_ndarray[index, slice])
                label_tensor[3, :, :, slice] = np.rot90(labels_ndarray[index, slice])
            # ============================================================================

            feed_dict ={phInput: input_tensor, phLabel: label_tensor}

            # Setting the learning rate
            if epoch <= 30:
                eta_val = 0.0001
                feed_dict.update({phEta: eta_val})
                _, loss_val, mse_val = sess.run([opt, loss, mse_loss], feed_dict=feed_dict)
            else:
                eta_val = 0.00001
                feed_dict.update({phEta: eta_val})
                _, loss_val, mse_val = sess.run([opt, loss, mse_loss], feed_dict=feed_dict)


            Training_Loss += loss_val*255.*255.
            MSE_Training_Loss += mse_val*255.*255.
        # end of one epoch

        # log ...
        Mean_Training_Loss = Training_Loss/N_samples
        Mean_MSE_Training_Loss = MSE_Training_Loss/N_samples

        Training_Loss = 0
        MSE_Training_Loss = 0

        epoch_time = time.time() - start_time
        msg = "Epoch: %d, Mean MSE: %.2f, Mean Loss: %.2f, Time: %.2f"
        msg_tuple = (epoch, Mean_MSE_Training_Loss, Mean_Training_Loss,epoch_time)
        print(msg%msg_tuple)

        target=open("%s/scores_training.txt" % out_folder, 'a')
        msg = "%d\t%.2f\t%.2f\t%d\n"
        target.write(msg%msg_tuple)
        target.close()

        # Save the trained model
        #   We will save it after each epoch in a separate folder.
        os.makedirs("%s/%04d" % (out_folder, epoch))
        saver.save(sess, "%s/%04d/model.ckpt" % (out_folder, epoch))
        saver.save(sess,"%s/model.ckpt"%out_folder) # save the last model in the root folder

        # ************ VALIDATION
        # Unfortunately, we do not have lots of training data. So,
        # we use the test data as validation data set.
        if epoch in see_valid_imgs:

            MSE_Validation_Loss = 0
            counter = 0
            start_time = time.time()

            for ind in range(0,len(val_inp_fnames),N_blks):
                test = np.float32(cv2.imread(val_inp_fnames[ind], cv2.IMREAD_GRAYSCALE)) / 255.0
                h, w = test.shape
                input_tensor = np.zeros((1, h, w, N_blks))
                input_tensor[0, :, :, 0] = test
                input_tensor[0, :, :, 1] = np.float32(cv2.imread(val_inp_fnames[ind + 1], cv2.IMREAD_GRAYSCALE)) / 255.0
                input_tensor[0, :, :, 2] = np.float32(cv2.imread(val_inp_fnames[ind + 2], cv2.IMREAD_GRAYSCALE)) / 255.0
                input_tensor[0, :, :, 3] = np.float32(cv2.imread(val_inp_fnames[ind + 3], cv2.IMREAD_GRAYSCALE)) / 255.0
                input_tensor[0, :, :, 4] = np.float32(cv2.imread(val_inp_fnames[ind + 4], cv2.IMREAD_GRAYSCALE)) / 255.0

                label_tensor = np.zeros((1, h, w, N_blks))
                label_tensor[0, :, :, 0] = np.float32(cv2.imread(val_lbl_fnames[counter], cv2.IMREAD_GRAYSCALE)) / 255.0
                counter+=1

                net_out_tensor = sess.run(net_out, feed_dict={phInput: input_tensor})
                im_out = net_out_tensor[0, :, :, 5]

                im_out = np.minimum(np.maximum(im_out, 0.0), 1.0) * 255.0
                im_out = np.uint8(im_out)

                cv2.imwrite("%s/%04d/%d.tif"%(out_folder,epoch,int(ind/N_blks)+1),im_out)

                mse_val = sess.run(mse_loss, feed_dict={phInput: input_tensor,
                                                        phLabel: label_tensor})
                MSE_Validation_Loss += mse_val * 255. * 255.
            # end of reconstructions

            # log ...
            Mean_reconst_time = (time.time() - start_time) / N_test
            Mean_MSE_Validation_Loss = MSE_Validation_Loss/N_test
            msg_tuple = (epoch,Mean_MSE_Validation_Loss, Mean_reconst_time)
            msg = "Test performance at epoch %d: , MSE: %.2f, Mean time: %.2f"
            print(msg%msg_tuple)

            target = open("%s/scores_test.txt" % out_folder, 'a')
            msg = "%d\t%.2f\t%.2f\n"
            target.write(msg%msg_tuple)
            target.close()
    sess.close()

# **********************************
# ************ TEST ****************
# **********************************

# The latest model is loaded at this point. So, we just need to
# feed it with test images.
# Some statistics related to the reconstruction of the test images
# will be saved in `test_scores.txt` in the `FINAL_RESULTS` folder.
if is_training==False:
    if os.path.isdir("%s/FINAL_RESULTS/" % (out_folder)) == False:
        os.makedirs("%s/FINAL_RESULTS/" % (out_folder))

    MSE_Validation_Loss = 0
    counter = 0
    start_time = time.time()

    for ind in range(0, len(val_inp_fnames), N_blks):

        print('Reconstructing test image #%d/%d'%((counter+1),N_test))
        test = np.float32(cv2.imread(val_inp_fnames[ind], cv2.IMREAD_GRAYSCALE)) / 255.0
        h, w = test.shape
        input_tensor = np.zeros((1, h, w, N_blks))
        input_tensor[0, :, :, 0] = test
        input_tensor[0, :, :, 1] = np.float32(cv2.imread(val_inp_fnames[ind + 1], cv2.IMREAD_GRAYSCALE)) / 255.0
        input_tensor[0, :, :, 2] = np.float32(cv2.imread(val_inp_fnames[ind + 2], cv2.IMREAD_GRAYSCALE)) / 255.0
        input_tensor[0, :, :, 3] = np.float32(cv2.imread(val_inp_fnames[ind + 3], cv2.IMREAD_GRAYSCALE)) / 255.0
        input_tensor[0, :, :, 4] = np.float32(cv2.imread(val_inp_fnames[ind + 4], cv2.IMREAD_GRAYSCALE)) / 255.0

        label_tensor = np.zeros((1, h, w, N_blks))
        label_tensor[0, :, :, 0] = np.float32(cv2.imread(val_lbl_fnames[counter], cv2.IMREAD_GRAYSCALE)) / 255.0
        counter += 1

        net_out_tensor = sess.run(net_out, feed_dict={phInput: input_tensor})
        im_out = net_out_tensor[0, :, :, 5]

        im_out = np.minimum(np.maximum(im_out, 0.0), 1.0) * 255.0
        im_out = np.uint8(im_out)

        cv2.imwrite("%s/FINAL_RESULTS/%d.tif" % (out_folder, int(ind / N_blks) + 1), im_out)

        mse_val = sess.run(mse_loss, feed_dict={phInput: input_tensor,
                                                phLabel: label_tensor})
        MSE_Validation_Loss += mse_val * 255. * 255.
    # end of reconstructions

    # log ...
    Mean_reconst_time = (time.time() - start_time) / N_test
    Mean_MSE_Validation_Loss = MSE_Validation_Loss / N_test
    msg_tuple = (Mean_MSE_Validation_Loss, Mean_reconst_time)
    msg = "MSE: %.2f, Mean time: %.2f"
    print(msg % msg_tuple)

    target = open("%s/FINAL_RESULTS/scores_test.txt" % out_folder, 'a')
    msg = "%.2f\t%.2f\n"
    target.write(msg % msg_tuple)
    target.close()
    sess.close()