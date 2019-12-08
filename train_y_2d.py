import os
import sys
import numpy as np
import tensorflow as tf
from load_2d_dataset import load_2d_dataset
from network import neural_network
from data_sampler_2d import sample_2d_dataset
from matplotlib import pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataset = load_2d_dataset()


flag = int(sys.argv[1])
run_num = int(sys.argv[2])

# dataset = np.repeat(dataset, 10, axis=0)

print(dataset.shape)

epochs = 10
batch_size = 1024
lr = 0.01
N = 2

data_sampler = sample_2d_dataset(dataset, batch_size, N)
n_batches = int(len(dataset)/batch_size)

zero_arr = np.zeros([batch_size, 1])

x_arr = np.around(np.arange(-1, 1.001, 0.01), decimals=2)
res_img = np.zeros([x_arr.shape[0], x_arr.shape[0]])


g = tf.Graph()

with g.as_default():
    model_G = neural_network(2,1,[10], name='Model_G_')
    init_g = tf.initialize_all_variables()
    x_g = tf.placeholder(tf.float64, [None, 2])
    G = model_G.value(x_g)
    d1g = model_G.dx(x_g)
    d2g = model_G.d2x(x_g)


d = tf.Graph()

with d.as_default():
    model_D = neural_network(2,1,[10, 10], name='Model_D_')
    init_d = tf.initialize_all_variables()
    x_d = tf.placeholder(tf.float64, [None, 2])
    D = model_D.value(x_d)
    d1d = model_D.dx(x_d)
    d2d = model_D.d2x(x_d)


yl = tf.Graph()

with yl.as_default():
    model_Y = neural_network(2, 1, [10, 10, 10, 10, 10], name='Model_Y_')
    x = tf.placeholder(tf.float64, [None, 2])
    y = tf.placeholder(tf.float64, [None, 1])
    v_d2g = tf.placeholder(tf.float64, [None, 2])
    v_d = tf.placeholder(tf.float64, [None, 1])
    ret = tf.placeholder(tf.float64, [None, 1])
    v_d2d = tf.placeholder(tf.float64, [None, 2])
    v_d1d = tf.placeholder(tf.float64, [None, 2])
    Y = model_Y.value(x)
    d1y = model_Y.dx(x)
    d2y = model_Y.d2x(x)
    lu = model_Y.compute_lu(x, v_d2g, v_d, v_d2d, v_d1d, ret)
    network_out = model_Y.value(x)
    loss = tf.reduce_mean(tf.nn.l2_loss(lu - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    init_y = tf.initialize_all_variables()


with tf.Session(graph=yl) as sess1:
    sess1.run(init_y)
    if flag:
        saver = tf.train.Saver()
        path = saver.restore(sess1, "./Models_sq/Model_Y/model_Y_last_epoch.ckpt")
    best_loss = sys.maxsize

    for e in range(1,epochs+1):
        epoch_loss = 0
        for b in range(n_batches):
            batch_x, batch_y = data_sampler.get_new_sample()

            with tf.Session(graph=g) as sess:
                sess.run(init_g)
                saver = tf.train.Saver()
                path = saver.restore(sess, "./Models_sq/Model_G/model_G.ckpt")
                [G_val, D1G, D2G] = sess.run([G, d1g, d2g], feed_dict={x_g: batch_x})

            with tf.Session(graph=d) as sess:
                sess.run(init_d)
                saver = tf.train.Saver()
                path = saver.restore(sess, "./Models_sq/Model_D/model_D.ckpt")
                [D_val, D1D, D2D] = sess.run([D, d1d, d2d], feed_dict={x_d: batch_x})

            _, l = sess1.run([optimizer, loss], feed_dict={x: batch_x,
                                                          y: batch_y,
                                                          v_d2g : D2G,
                                                          v_d: D_val,
                                                          v_d1d : D1D,
                                                          v_d2d : D2D,
                                                          ret : zero_arr
                                                          })

            epoch_loss += l / n_batches

            print(f"=> Done with {b + 1} / {n_batches}  Batch Loss: {l:.6f}")

        print(f"===> Epoch {e} Complete: Avg. Training Loss: {epoch_loss:.6f}")

        saver = tf.train.Saver()
        path = saver.save(sess1, "./Models_sq/Model_Y/model_Y_last_epoch.ckpt")

        with tf.Session(graph=g) as sess:
            sess.run(init_g)
            saver = tf.train.Saver()
            path = saver.restore(sess, "./Models_sq/Model_G/model_G.ckpt")
            G_val = sess.run(G, feed_dict={x_g: dataset[:, 0:2].reshape((dataset.shape[0], 2))})

        with tf.Session(graph=d) as sess:
            sess.run(init_d)
            saver = tf.train.Saver()
            path = saver.restore(sess, "./Models_sq/Model_D/model_D.ckpt")
            D_val = sess.run(D, feed_dict={x_d: dataset[:, 0:2].reshape((dataset.shape[0], 2))})

        Y_val = sess1.run(Y, feed_dict={x: dataset[:, 0:2].reshape((dataset.shape[0], 2))})

        y_pred = G_val + D_val * Y_val

        for it in range(dataset.shape[0]):
            idx_x = np.where(x_arr == dataset[it, 0])
            idx_y = np.where(x_arr == dataset[it, 1])
            res_img[idx_x, idx_y] = y_pred[it]

        plt.figure()
        plt.imshow(res_img)
        plt.colorbar()
        plt.savefig('./Plots_Test/U_res_sq.png')
        plt.close()

        saver = tf.train.Saver()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            path = saver.save(sess1, "./Models_sq/Model_Y/model_Y_best" + str(run_num) + ".ckpt")

# plt.figure(2); plt.plot(dataset[:,0].reshape((dataset.shape[0],1)),D_pred,dataset[:,0].reshape((dataset.shape[0],1)),G_pred)
# plt.savefig('./Plots/2.png')

print("All Done")