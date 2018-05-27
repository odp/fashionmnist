# https://henning.kropponline.de/2017/03/19/distributing-tensorflow/

import argparse
import sys
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.keras._impl import keras

import tensorflow as tf

FLAGS = None

input_shape = [None, 28, 28, 1]
number_of_classes = 10

#Hyper parameters
batch_size = 128

def model_fn(input_shape, number_of_classes):
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(1e-6, global_step, 1000, 0.96, staircase=True)

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
    
    #training operartion
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    
    #accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))
    
    return { "logits": logits,
             "predictions": predictions,
             "loss": loss,
             "train_op": train_op,
             "accuracy": accuracy,
             "x": input_layer,
             "y": labels,
             "train_mode": train_mode }

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        with tf.device(tf.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % FLAGS.task_index,
                    cluster=cluster)):

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

        def run_train_epoch(target,FLAGS,epoch_index):
            epoch_cost, epoch_accuracy = 0, 0
            with tf.train.MonitoredTrainingSession(master=target,
                                                   is_chief=(FLAGS.task_index == 0),
                                                   checkpoint_dir=FLAGS.logs_dir) as sess:

                    total_size = x_train.shape[0]
                    number_of_batches = int(total_size/batch_size)
                   
                    for i in range(number_of_batches):
                        mini_x = x_train[i*batch_size:(i+1)*batch_size, :, :, :]
                        mini_y = y_train[i*batch_size:(i+1)*batch_size, :]
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
                    print("Epoch: {} Cost: {} accuracy: {} ".format(epoch_index+1, np.squeeze(epoch_cost), epoch_accuracy))
    

        for e in range(FLAGS.nb_epochs):
            run_train_epoch(server.target, FLAGS, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--nb_epochs",
        type=int,
        default=5,
        help="Number of epochs?"
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="./logs",
        help="Logs directory"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


