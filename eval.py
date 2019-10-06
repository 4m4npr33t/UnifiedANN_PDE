import os
from load_dataset import load_test_dataset
import tensorflow as tf
from matplotlib import pyplot as plt
from network import neural_network
import numpy as np


os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataset = load_test_dataset()


plt.figure(1); plt.scatter(dataset[:,0].reshape((dataset.shape[0],1)), dataset[:,2].reshape((dataset.shape[0],1)))
plt.savefig('./Plots/u.png')

plt.figure(2); plt.scatter(dataset[:,0].reshape((dataset.shape[0],1)), dataset[:,3].reshape((dataset.shape[0],1)))
plt.savefig('./Plots/g.png')

plt.figure(5); plt.scatter(dataset[:,0].reshape((dataset.shape[0],1)), dataset[:,4].reshape((dataset.shape[0],1)))
plt.savefig('./Plots/d.png')

print(dataset.shape)


g = tf.Graph()

with g.as_default():
    model_G = neural_network(1,1,[10], name='Model_G_')
    init_g = tf.initialize_all_variables()
    x_g = tf.placeholder(tf.float64, [None, 1])
    G = model_G.value(x_g)
    d1g = model_G.dx(x_g)
    d2g = model_G.d2x(x_g)


d = tf.Graph()

with d.as_default():
    model_D = neural_network(1,1,[10,10], name='Model_D_')
    init_d = tf.initialize_all_variables()
    x_d = tf.placeholder(tf.float64, [None, 1])
    D = model_D.value(x_d)
    d1d = model_D.dx(x_d)
    d2d = model_D.d2x(x_d)


yl = tf.Graph()

with yl.as_default():
    model_Y = neural_network(1,1,[10,10], name='Model_Y_')
    x = tf.placeholder(tf.float64, [None, 1])
    y = tf.placeholder(tf.float64, [None, 1])
    v_d2g = tf.placeholder(tf.float64, [None, 1])
    v_d = tf.placeholder(tf.float64, [None, 1])
    v_d2d = tf.placeholder(tf.float64, [None, 1])
    v_d1d = tf.placeholder(tf.float64, [None, 1])
    Y = model_Y.value(x)
    d1y = model_Y.dx(x)
    d2y = model_Y.d2x(x)
    lu = model_Y.compute_lu(x, v_d2g, v_d, v_d2d, v_d1d)
    network_out = model_Y.value(x)
    init_y = tf.initialize_all_variables()


with tf.Session(graph=yl) as sess1:
    sess1.run(init_y)
    saver = tf.train.Saver()
    path = saver.restore(sess1, "./Models/Model_Y/model_Y_small_lr.ckpt")
    with tf.Session(graph=g) as sess:
        sess.run(init_g)
        saver = tf.train.Saver()
        path = saver.restore(sess, "./Models/Model_G/model_G.ckpt")
        G_val = sess.run(G, feed_dict={x_g: dataset[:, 0].reshape((dataset.shape[0], 1))})

    with tf.Session(graph=d) as sess:
        sess.run(init_d)
        saver = tf.train.Saver()
        path = saver.restore(sess, "./Models/Model_D/model_D.ckpt")
        D_val = sess.run(D, feed_dict={x_d: dataset[:, 0].reshape((dataset.shape[0], 1))})

    Y_val = sess1.run(network_out, feed_dict={x: dataset[:, 0].reshape((dataset.shape[0], 1))})

    pred = G_val + D_val * Y_val
    print(np.min(pred-dataset[:,2].reshape((dataset.shape[0],1))))

    plt.figure(3);
    plt.scatter(dataset[:, 0].reshape((dataset.shape[0], 1)), G_val)
    plt.savefig('./Plots/g_hat.png')

    plt.figure(4);
    plt.scatter(dataset[:, 0].reshape((dataset.shape[0], 1)), pred)
    plt.savefig('./Plots/U_hat.png')

    plt.figure(6);
    plt.scatter(dataset[:, 0].reshape((dataset.shape[0], 1)), D_val)
    plt.savefig('./Plots/d_hat.png')

    plt.figure(7);
    plt.scatter(dataset[:, 0].reshape((dataset.shape[0], 1)), D_val-dataset[:,4].reshape((dataset.shape[0],1)))
    plt.savefig('./Plots/d_diff.png')

    plt.figure(8);
    plt.scatter(dataset[:, 0].reshape((dataset.shape[0], 1)), G_val-dataset[:,3].reshape((dataset.shape[0],1)))
    plt.savefig('./Plots/g_diff.png')

    plt.figure(9);
    plt.scatter(dataset[:, 0].reshape((dataset.shape[0], 1)), pred-dataset[:,2].reshape((dataset.shape[0],1)))
    plt.savefig('./Plots/u_diff.png')



