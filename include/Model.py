import tensorflow as tf
import numpy as np
import math, sys
from tensorflow.python.client import device_lib
# from __future__ import absolute_import, division, print_function, unicode_literals

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()

def get_available_gpus():
    local_device_protos=device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def cnn_model(images,labels,
              val_images,val_labels,
              n_epochs,batch_size,
              learning_rate,
              save_model_path):

    # gpu_options=tf.GPUOptions(allow_growth=True)

    # Check device
    device=get_available_gpus()
    if len(device)>1:
        multi_gpu=True
    elif len(device)==1:
        multi_gpu=False

    # 3D images
    size=images.shape
    H, W, T, C=size[1], size[2], size[3], 1
    shape=[-1,H,W,T,C]

    iterate=3
    num_examples=size[0]

    conv_fmaps=4
    conv_ksize=3
    conv_stride=1
    conv_pad="SAME"

    pool_ksize=3
    pool_stride=1
    pool_pad="VALID"

    n_fc1=2048
    n_fc2=256
    n_outputs=1

    dp_rate1=0.5

    reset_graph()

    # Input layer
    with tf.name_scope("inputs"):
        X=tf.placeholder(tf.float32,shape=[None,H,W,T,1],name="X")
        X_reshaped=tf.reshape(X,shape=shape)
        Y=tf.placeholder(tf.int32,shape=[None,1],name="Y")

    # Convolution Layer
    conv=tf.layers.conv3d(X_reshaped,
                          filters=conv_fmaps,
                          kernel_size=conv_ksize,
                          strides=conv_stride,
                          padding=conv_pad,
                          activation=tf.nn.relu,
                          name="conv1")

    pool=tf.nn.max_pool3d(conv,
                          ksize=[1,pool_ksize,pool_ksize,pool_ksize,1],
                          strides=[1,pool_ksize,pool_ksize,pool_ksize,1],
                          padding=pool_pad,
                          name="pool1")

    with tf.name_scope("norm1"):
        norm=tf.contrib.layers.batch_norm(pool,
                                          data_format='NHWC',
                                          center=True,
                                          scale=True)

    """
    norm=tf.nn.batch_normalization(pool,
                                   name="norm1")
    """

    for i in range(iterate):
        times=int((i+1)/2)
        conv=tf.layers.conv3d(norm,
                              filters=conv_fmaps*2**times,
                              kernel_size=conv_ksize*2**times,
                              strides=conv_stride,
                              padding=conv_pad,
                              activation=tf.nn.relu,
                              name="conv"+str(i+2))
        temp_pool_ksize=pool_ksize+2*times
        if (i+1)%2 == 0:
            pool=tf.nn.max_pool3d(conv,
                                  ksize=[1,temp_pool_ksize,temp_pool_ksize,temp_pool_ksize,1],
                                  strides=[1,pool_ksize,pool_ksize,pool_ksize,1],
                                  padding=pool_pad,
                                  name="pool"+str(i+2))
        with tf.name_scope("norm"+str(i+2)):
            norm=tf.contrib.layers.batch_norm(pool,
                                              data_format='NHWC',
                                              center=True,
                                              scale=True)
        """
        norm=tf.nn.batch_normalization(pool,
                                       name="norm"+str(i+2))
        """

    norm_flat=tf.contrib.layers.flatten(norm)

    # Dense Layer
    fc=tf.layers.dense(norm_flat,
                       n_fc1,
                       activation=tf.nn.relu,
                       name="fc1")
    fc=tf.layers.dropout(fc,
                         rate=dp_rate1,
                         name="dropout1")
    fc=tf.layers.dense(fc,
                       n_fc2,
                       activation=tf.nn.relu,
                       name="fc2")

    # Output Layer
    with tf.name_scope("output"):
        logits=tf.layers.dense(fc,
                               n_outputs,
                               name="output")
        Y_proba=tf.nn.sigmoid(logits,
                              name="Y_proba")
        tf.add_to_collection("pred_network",Y_proba)

    # Setting of Training
    with tf.name_scope("train"):
        Y=tf.cast(Y,tf.float32)
        xentropy=tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=Y_proba)
        # xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=Y)
        loss=tf.reduce_mean(xentropy)
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_op=optimizer.minimize(loss)

    # Match Answers
    with tf.name_scope("eval"):
        # Y=tf.cast(Y,tf.int32)
        # Y=tf.reshape(Y,[-1])
        error=tf.abs(Y-Y_proba)
        # accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
        accuracy=tf.ones([1],dtype=tf.float32)-tf.reduce_mean(error)
        xentropy=tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=Y_proba)
        loss=tf.reduce_mean(xentropy)

    with tf.name_scope("init_and_save"):
        init=tf.global_variables_initializer()
        saver=tf.train.Saver()

    # Record
    Record={}
    Record['acc_train']=[]
    Record['loss_train']=[]
    Record['acc_val']=[]
    Record['loss_val']=[]

    # Training
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            acc_train,loss_train=0,0
            acc_val,loss_val=0,0
            num_iter=num_examples // batch_size

            for iteration in range(num_iter):
                #this cycle is for dividing step by step the heavy work of each neuron
                if iteration+1 < num_iter:
                    X_batch=images[iteration*batch_size:iteration*batch_size+batch_size,...]
                    Y_batch=labels[iteration*batch_size:iteration*batch_size+batch_size,...]
                    sess.run(training_op,feed_dict={X: X_batch, Y: Y_batch})
                    view_bar("processing image of " ,(iteration+1)*batch_size, num_examples)
                    acc_train=acc_train+accuracy.eval(feed_dict={X: X_batch, Y: Y_batch})*batch_size
                    loss_train=loss_train+loss.eval(feed_dict={X: X_batch, Y: Y_batch})*batch_size
                else:
                    X_batch=images[iteration*batch_size:,...]
                    Y_batch=labels[iteration*batch_size:,...]
                    sess.run(training_op,feed_dict={X: X_batch, Y: Y_batch})
                    view_bar("processing image of " , num_examples, num_examples)
                    acc_train=acc_train+accuracy.eval(feed_dict={X: X_batch, Y: Y_batch})*(num_examples-iteration*batch_size)
                    loss_train=loss_train+loss.eval(feed_dict={X: X_batch, Y: Y_batch})*(num_examples-iteration*batch_size)
            acc_train=acc_train/num_examples
            loss_train=loss_train/num_examples

            if device:
                num_val=val_images.shape[0]
                val_num_iter=num_val // batch_size
                with tf.device(device[-1]):
                    for iteration in range(val_num_iter):
                        if iteration+1 < val_num_iter:
                            val_X_batch=val_images[iteration*batch_size:iteration*batch_size+batch_size,...]
                            val_Y_batch=val_labels[iteration*batch_size:iteration*batch_size+batch_size,...]
                            acc_val=acc_val+accuracy.eval(feed_dict={X: val_X_batch, Y: val_Y_batch})*batch_size
                            loss_val=loss_val+loss.eval(feed_dict={X: val_X_batch, Y: val_Y_batch})*batch_size
                        else:
                            val_X_batch=val_images[iteration*batch_size:,...]
                            val_Y_batch=val_labels[iteration*batch_size:,...]
                            acc_val=acc_val+accuracy.eval(feed_dict={X: val_X_batch, Y: val_Y_batch})*(num_val-iteration*batch_size)
                            loss_val=loss_val+loss.eval(feed_dict={X: val_X_batch, Y: val_Y_batch})*(num_val-iteration*batch_size)
                    acc_val=acc_val/num_val
                    loss_val=loss_val/num_val

            Record['acc_train'].append(acc_train)
            Record['loss_train'].append(loss_train)
            Record['acc_val'].append(acc_val)
            Record['loss_val'].append(loss_val)

            print("\nEpoch:",epoch+1, "Train accuracy:", acc_train, "Train loss:", loss_train, "Validation accuracy:", acc_val, "Validation loss:", loss_val)

        saver.save(sess,save_model_path)

    return Record

def predict(images,model_path,par_path):
    sess=tf.Session()
    saver=tf.train.import_meta_graph(model_path)
    saver.restore(sess,tf.train.latest_checkpoint(par_path))
    graph=tf.get_default_graph()
    X=graph.get_operation_by_name('inputs/X').outputs[0]
    Y=graph.get_collection('pred_network')[0]
    ans=sess.run(Y,feed_dict={X: images})

    return ans

