import numpy as np
import tensorflow as tf
from network import neural_network
from load_2d_dataset import load_2d_dataset
from matplotlib import pyplot as plt


dataset = load_2d_dataset()

x_arr = np.arange(-1, 1.001, 0.01)
res_img = np.zeros([x_arr.shape[0], x_arr.shape[0]])

g = tf.Graph()

with g.as_default():
    model_G = neural_network(2,1,[10], name='Model_G_')
    init_g = tf.initialize_all_variables()
    x_g = tf.placeholder(tf.float64, [None, 2])
    G = model_G.value(x_g)


d = tf.Graph()

with d.as_default():
    model_D = neural_network(2,1,[10], name='Model_D_')
    init_d = tf.initialize_all_variables()
    x_d = tf.placeholder(tf.float64, [None, 2])
    D = model_D.value(x_d)


yl = tf.Graph()

with yl.as_default():
    model_Y = neural_network(2,1,[10, 10, 10, 10, 10], name='Model_Y_')
    x = tf.placeholder(tf.float64, [None, 2])
    Y = model_Y.value(x)


with tf.Session(graph=yl) as sess1:
    saver = tf.train.Saver()
    path = saver.restore(sess1, "./Models_2d/Model_Y/model_Y.ckpt")
    with tf.Session(graph=g) as sess:
        sess.run(init_g)
        saver = tf.train.Saver()
        path = saver.restore(sess, "./Models_2d/Model_G/model_G.ckpt")
        G_val = sess.run(G, feed_dict={x_g: dataset[:, 0:2].reshape((dataset.shape[0], 2))})

    with tf.Session(graph=d) as sess:
        sess.run(init_d)
        saver = tf.train.Saver()
        path = saver.restore(sess, "./Models_2d/Model_D/model_D.ckpt")
        D_val = sess.run(D, feed_dict={x_d: dataset[:, 0:2].reshape((dataset.shape[0], 2))})

    Y_val = sess1.run(Y, feed_dict={x: dataset[:, 0:2].reshape((dataset.shape[0], 2))})

    y_pred = G_val + D_val*Y_val

for it in range(dataset.shape[0]):
    idx_x = np.where(x_arr == dataset[it, 0])
    idx_y = np.where(x_arr == dataset[it, 1])
    res_img[idx_x, idx_y] = y_pred[it]


plt.figure(1)
plt.imshow(res_img)
plt.colorbar()
plt.savefig('./Plots_2d/U_res.png')

print("All Done")