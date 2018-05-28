# references
# 1. https://clusterone.com/blog/2017/09/13/distributed-tensorflow-clusterone/
# 2. https://henning.kropponline.de/2017/03/19/distributing-tensorflow/

import argparse
import sys
import numpy as np
import logging
import tensorflow as tf
import os

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.keras._impl import keras

LOCAL_LOG_LOCATION = "./logs"

from clusterone import get_logs_path

#Create logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

input_shape = [None, 28, 28, 1]
number_of_classes = 10
learning_rate = 0.01

#Hyper parameters
batch_size = 128

def model_fn(input_shape, number_of_classes):

    input_layer = tf.placeholder(tf.float32, shape=input_shape)
    labels = tf.placeholder(tf.float32, shape=[None, number_of_classes])
    train_mode = tf.placeholder(tf.bool)

    #image ->conv2d->maxpooling->conv2d->maxpooling->flatten->dense->dropout->logits
    # initialize random seed
    tf.set_random_seed(1)
    
    #convolution layer 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer, 
        filters=32, 
        kernel_size=[5, 5], 
        padding="same", 
        activation=tf.nn.relu)
    
    #pooling layer 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    
    #convolution layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1, 
        filters=64, 
        kernel_size=[5, 5], 
        padding="same", 
        activation=tf.nn.relu)
    
    #pooling layer 1
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    #flatten the output volume of pool2 into a vector
    pool2_flat = tf.reshape(pool2, shape=[-1, 7*7*64])
    
    #dense layer
    dense = tf.layers.dense(
        inputs=pool2_flat, 
        units=1024,
        activation=tf.nn.relu)
    
    #dropout regularization
    dropout = tf.layers.dropout(
        inputs=dense, 
        rate=0.2, 
        training=train_mode)
    
    #logits layer
    logits = tf.layers.dense(inputs=dropout, units=10)
    
    predictions = {
        "classes" : tf.argmax(input=logits, axis=1),
        "probabilities" : tf.nn.softmax(logits=logits)
    }
    
    #loss
    loss = tf.losses.softmax_cross_entropy(labels, logits)
        
    training_summary = tf.summary.scalar('Training_Loss', loss)
    global_step = tf.train.get_or_create_global_step()
    
    #training operartion
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    
    #accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))
    
    acc_summary = tf.summary.scalar('accuracy', accuracy)
    
    return { "logits": logits,
             "predictions": predictions,
             "loss": loss,
             "train_op": train_op,
             "accuracy": accuracy,
             "x": input_layer,
             "y": labels,
             "train_mode": train_mode }

def main():
    # clusterone snippet 1 - get environment variables
    try:
        job_name = os.environ['JOB_NAME']
        task_index = os.environ['TASK_INDEX']
        ps_hosts = os.environ['PS_HOSTS']
        worker_hosts = os.environ['WORKER_HOSTS']
    except:
        job_name = None
        task_index = 0
        ps_hosts = None
        worker_hosts = None

    #end of clusterone snippet 1

    #Flags
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    PATH_TO_LOCAL_LOGS = os.path.expanduser(LOCAL_LOG_LOCATION)
    
    # clusterone snippet 2: flags.
    flags.DEFINE_string("logs_dir",
        get_logs_path(root=PATH_TO_LOCAL_LOGS),
        "Path to store logs and checkpoints")

    # Define worker specific environment variables. Handled automatically.
    flags.DEFINE_string("job_name", job_name,
                        "job name: worker or ps")
    flags.DEFINE_integer("task_index", task_index,
                        "Worker task index, should be >= 0. task_index=0 is "
                        "the chief worker task the performs the variable "
                        "initialization")
    flags.DEFINE_string("ps_hosts", ps_hosts,
                        "Comma-separated list of hostname:port pairs")
    flags.DEFINE_string("worker_hosts", worker_hosts,
                        "Comma-separated list of hostname:port pairs")

    # end of clusterone snippet 2

    # Training flags - feel free to play with that!
    flags.DEFINE_integer("batch", 64, "Batch size")
    flags.DEFINE_integer("time", 1, "Number of frames per sample")
    flags.DEFINE_integer("steps_per_epoch", 10000, "Number of training steps per epoch")
    flags.DEFINE_integer("nb_epochs", 200, "Number of epochs")


    # Model flags - feel free to play with that!
    flags.DEFINE_float("dropout_rate1",.2,"Dropout rate on first dropout layer")
    flags.DEFINE_float("dropout_rate2",.5,"Dropout rate on second dropout layer")
    flags.DEFINE_float("starter_lr",1e-6,"Starter learning rate. Exponential decay is applied")
    flags.DEFINE_integer("fc_dim",512,"Size of the dense layer")
    flags.DEFINE_boolean("nogood",False,"Ignore `goods` filters.")


    # clusterone snippet 3: configure distributed environment
    def device_and_target():
        # If FLAGS.job_name is not set, we're running single-machine TensorFlow.
        # Don't set a device.
        if FLAGS.job_name is None:
            print("Running single-machine training")
            return (None, "")

        # Otherwise we're running distributed TensorFlow.
        print("Running distributed training")
        if FLAGS.task_index is None or FLAGS.task_index == "":
            raise ValueError("Must specify an explicit `task_index`")
        if FLAGS.ps_hosts is None or FLAGS.ps_hosts == "":
            raise ValueError("Must specify an explicit `ps_hosts`")
        if FLAGS.worker_hosts is None or FLAGS.worker_hosts == "":
            raise ValueError("Must specify an explicit `worker_hosts`")

        cluster_spec = tf.train.ClusterSpec({
                "ps": FLAGS.ps_hosts.split(","),
                "worker": FLAGS.worker_hosts.split(","),
        })

        server = tf.train.Server(
                cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
        if FLAGS.job_name == "ps":
            server.join()

        worker_device = "/job:worker/task:{}".format(FLAGS.task_index)
        
        # The device setter will automatically place Variables ops on separate
        # parameter servers (ps). The non-Variable ops will be placed on the workers.
        return (tf.train.replica_device_setter(
                        worker_device=worker_device,
                        cluster=cluster_spec), server.target)

    device, target = device_and_target()
    # end of clusterone snippet 3
    
    if FLAGS.logs_dir is None or FLAGS.logs_dir == "":
        raise ValueError("Must specify an explicit `logs_dir`")

    with tf.device(device):
        with tf.name_scope("input"):
            (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

            x_train = x_train.astype('float32') / 255.
            x_test = x_test.astype('float32') / 255.
            
            x_train, x_valid = x_train[5000:], x_train[:5000]
            y_train, y_valid = y_train[5000:], y_train[:5000]

            # Reshape input data from (28, 28) to (28, 28, 1)
            w, h = 28, 28
            x_train = x_train.reshape(x_train.shape[0], w, h, 1)
            x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
            x_test = x_test.reshape(x_test.shape[0], w, h, 1)

            # One-hot encode the labels
            y_train = tf.keras.utils.to_categorical(y_train, 10)
            y_valid = tf.keras.utils.to_categorical(y_valid, 10)
            y_test = tf.keras.utils.to_categorical(y_test, 10)


        with tf.name_scope("model"):
            model = model_fn(input_shape, number_of_classes)
            x = model["x"]
            y = model["y"]
            train_mode = model["train_mode"]

    def shuffle(x, y):
        idxs = np.random.permutation(x.shape[0]) #shuffled ordering
        return x[idxs], y[idxs]
        
    def run_train_epoch(target,FLAGS,epoch_index):
        epoch_cost, epoch_accuracy = 0, 0
        x_train_r, y_train_r = shuffle(x_train, y_train)

        with tf.train.MonitoredTrainingSession(master=target, 
                is_chief=(FLAGS.task_index == 0), checkpoint_dir=FLAGS.logs_dir) as sess:
            total_size = x_train.shape[0]
            number_of_batches = int(total_size/batch_size)
           
            for i in range(number_of_batches):
                mini_x = x_train_r[i*batch_size:(i+1)*batch_size, :, :, :]
                mini_y = y_train_r[i*batch_size:(i+1)*batch_size, :] 
                _, cost = sess.run([model["train_op"], model["loss"]], 
                                    feed_dict={x:mini_x, y:mini_y, train_mode:True})

                train_accuracy = sess.run(model["accuracy"], 
                                          feed_dict={x:mini_x, y:mini_y, train_mode:False})
                epoch_cost += cost
                epoch_accuracy += train_accuracy

            epoch_cost /= number_of_batches

            if total_size % batch_size != 0:
                epoch_accuracy /= (number_of_batches+1)
            else:
                epoch_accuracy /= number_of_batches

            print("Epoch: {} Cost: {} accuracy: {} ".format(epoch_index+1, 
                np.squeeze(epoch_cost), epoch_accuracy))


    for e in range(FLAGS.nb_epochs):
        run_train_epoch(target, FLAGS, e)


if __name__ == "__main__":
    main()

