# references
# 1. https://clusterone.com/blog/2017/09/13/distributed-tensorflow-clusterone/
# 2. https://henning.kropponline.de/2017/03/19/distributing-tensorflow/

# depends on tensorflow >= 1.7

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

    # image ->conv2d->maxpooling->conv2d->maxpooling->flatten->dense->dropout->logits
    # initialize random seed
    tf.set_random_seed(1)
    
    #convolution layer 1 
    conv1 = tf.layers.conv2d(
        inputs=input_layer, 
        filters=32, 
        kernel_size=[5, 5], 
        padding="same", 
        activation=tf.nn.relu)
    
    print(conv1.get_shape()) # (?, 28, 28, 64)
    
    #pooling layer 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    
    print(pool1.get_shape()) # (?, 14, 14, 64)
    
    #convolution layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1, 
        filters=64, 
        kernel_size=[5, 5], 
        padding="same", #  out_height = ceil(float(in_height) / float(strides[1]))
        activation=tf.nn.relu)
    
    print(conv2.get_shape()) # (?, 14, 14, 64)

    #pooling layer 1 
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    print(pool2.get_shape()) # (?, 7, 7, 64)

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
    loss = tf.losses.softmax_cross_entropy(labels, logits) # H(p, q) = -sum x: p(x) * log q(x)
        
    global_step = tf.train.get_or_create_global_step()
    
    #training operartion
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    
    #accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))
    
    tf.summary.scalar('train_loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    
    summary_op = tf.summary.merge_all()
    
    return { "logits": logits,
             "predictions": predictions,
             "loss": loss,
             "train_op": train_op,
             "accuracy": accuracy,
             "summary": summary_op,
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

    flags.DEFINE_integer("nb_epochs", 20, "Number of epochs")

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

        # Represents a cluster as a set of "tasks", organized into "jobs".
        cluster_spec = tf.train.ClusterSpec({
                "ps": FLAGS.ps_hosts.split(","), # job1
                "worker": FLAGS.worker_hosts.split(","), # job2
        })

        # Server instance encapsulates a set of devices and a tf.Session
        # target that can participate in distributed training. A server belongs
        # to a cluster (specified by a tf.train.ClusterSpec), and corresponds to
        # a particular task in a named job.
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

    device, target = device_and_target() # place tensors, session
    # end of clusterone snippet 3
    
    if FLAGS.logs_dir is None or FLAGS.logs_dir == "":
        raise ValueError("Must specify an explicit `logs_dir`")

    with tf.device(device):
        with tf.name_scope("input"):
            (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

            print(x_train.shape, x_test.shape) # 60k, 10k

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
        
    def run_train_epoch(target, FLAGS, epoch_index, train_writer, test_writer):
        epoch_loss, epoch_accuracy = 0, 0
        x_train_r, y_train_r = shuffle(x_train, y_train)

        with tf.train.MonitoredTrainingSession(master=target, 
                is_chief=(FLAGS.task_index == 0), checkpoint_dir=FLAGS.logs_dir) as sess:
            total_size = x_train.shape[0]
            number_of_batches = int(total_size/batch_size)
           
            for i in range(number_of_batches):
                step = epoch_index * number_of_batches + i

                mini_x = x_train_r[i*batch_size:(i+1)*batch_size, :, :, :]
                mini_y = y_train_r[i*batch_size:(i+1)*batch_size, :] 
                _, loss = sess.run([model["train_op"], model["loss"]],
                                    feed_dict={x:mini_x, y:mini_y, train_mode:True})

                epoch_loss += loss
                
                train_accuracy, summary = sess.run([model["accuracy"], model["summary"]],
                                          feed_dict={x:mini_x, y:mini_y, train_mode:False})
                
                epoch_accuracy += train_accuracy
                
                train_writer.add_summary(summary, step)

                if step % 200 == 0:  # Record summaries and test-set accuracy
                    test_accuracy, summary = sess.run([model["accuracy"], model["summary"]],
                                              feed_dict={x:x_test, y:y_test, train_mode:False})
                    test_writer.add_summary(summary, step)
                    print('test accuracy at step %s: %s' % (step, test_accuracy))

        epoch_loss /= number_of_batches
        epoch_accuracy /= number_of_batches

        print("Epoch: {} loss: {} train accuracy: {}".format(epoch_index+1, 
            np.squeeze(epoch_loss), epoch_accuracy))

        
    train_wr = tf.summary.FileWriter(FLAGS.logs_dir + '/train', graph=tf.get_default_graph())
    test_wr = tf.summary.FileWriter(FLAGS.logs_dir + '/test')

    for e in range(FLAGS.nb_epochs):
        run_train_epoch(target, FLAGS, e, train_wr, test_wr)


if __name__ == "__main__":
    main()

