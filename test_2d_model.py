import numpy as np
import tensorflow as tf
from network import neural_network
from load_2d_dataset import load_2d_dataset
from matplotlib import pyplot as plt


dataset = load_2d_dataset()

x_arr = np.around(np.arange(-1, 1.001, 0.01), decimals=2)
res_img = np.zeros([x_arr.shape[0], x_arr.shape[0]])
t_d = np.zeros([x_arr.shape[0], x_arr.shape[0]])
u_vals = np.zeros([x_arr.shape[0], x_arr.shape[0]])


x = tf.placeholder(tf.float64, [None, 2])
y = tf.placeholder(tf.float64, [None, 1])

model = neural_network(2,1,[10], name='Model_G_')

network_out = model.value(x)
init = tf.initialize_all_variables()

saver = tf.train.Saver(save_relative_paths=True)

with tf.Session() as sess:
    sess.run(init)

    path = saver.restore(sess, "./Models_sq/Model_G/model_G.ckpt")
    y_pred = sess.run(network_out, feed_dict={x: dataset[:, 0:2].reshape((dataset.shape[0], 2))})


for it in range(dataset.shape[0]):
    idx_x = np.where(x_arr == dataset[it, 0])
    idx_y = np.where(x_arr == dataset[it, 1])
    res_img[idx_x, idx_y] = y_pred[it]
    u_vals[idx_x, idx_y] = dataset[it, 3]


plt.figure(1)
plt.imshow(res_img)
plt.colorbar()
plt.savefig('./Plots_Test/G_res.png')

plt.figure()
plt.imshow(abs(res_img-u_vals))
plt.colorbar()
plt.savefig('./Plots_Test/diff_u_g.png')


plt.figure()
plt.imshow(u_vals)
plt.colorbar()
plt.savefig('./Plots_Test/True_Data.png')

print("All Done")