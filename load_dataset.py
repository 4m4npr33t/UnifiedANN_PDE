import numpy as np


def load_dataset():
    xi = np.load('./Data/pts.npy')
    f = np.load('./Data/f.npy')
    u = np.load('./Data/u.npy')
    g = np.load('./Data/g.npy')
    d = np.load('./Data/d.npy')

    data_set = np.concatenate((xi, f, u, g, d), axis=1)

    return data_set


def load_test_dataset():
    xi = np.load('./Data/pts_test.npy')
    f = np.load('./Data/f_test.npy')
    u = np.load('./Data/u_test.npy')
    g = np.load('./Data/g_test.npy')
    d = np.load('./Data/d_test.npy')

    data_set = np.concatenate((xi, f, u, g, d), axis=1)

    return data_set