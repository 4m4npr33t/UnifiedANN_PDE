import os
import sys
import numpy as np
import tensorflow as tf
from load_dataset import load_dataset
from network import neural_network
from data_sampler import sample_dataset
from matplotlib import pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES']='1'

dataset = load_dataset()

print(dataset.shape)

epochs = 2000
batch_size = 16
lr = 0.01
N = 3

data_sampler = sample_dataset(dataset, batch_size, N)
n_batches = int(len(dataset)/batch_size)


plt.figure(1); plt.scatter(dataset[:,0].reshape((dataset.shape[0],1)), dataset[:,N].reshape((dataset.shape[0],1)))
plt.savefig('./Plots/1.png')

#placeholders for training data

x = tf.placeholder(tf.float64, [None, 1])
y = tf.placeholder(tf.float64, [None, 1])

model = neural_network(1,1,[10], name='Model_G_')

network_out = model.value(x)

loss = tf.reduce_mean(tf.nn.l2_loss(network_out-y))

optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

init = tf.initialize_all_variables()

saver = tf.train.Saver(save_relative_paths=True)

with tf.Session() as sess:
    # create initialized variables
    best_loss = sys.maxsize
    sess.run(init)
    for e in range(1,epochs+1):
        epoch_loss = 0
        for b in range(n_batches):
            batch_x, batch_y = data_sampler.get_new_sample()
            _, l = sess.run([optimizer, loss], feed_dict={x: batch_x, y: batch_y})

            epoch_loss += l/n_batches

            print(f"=> Done with {b+1} / {n_batches}  Batch Loss: {l:.6f}")

        print(f"===> Epoch {e} Complete: Avg. Training Loss: {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            path = saver.save(sess, "./Models/Model_G/model_G.ckpt")
            print(path)

    y_pred = sess.run(network_out, feed_dict={x: dataset[:,0].reshape((dataset.shape[0],1))})

plt.figure(2); plt.scatter(dataset[:,0].reshape((dataset.shape[0],1)),y_pred)
plt.savefig('./Plots/2.png')

print(model.weights)

print("All Done")