import numpy as np


def load_2d_dataset():
    x = np.load('./Data_Square/x.npy')
    y = np.load('./Data_Square/y.npy')
    f = np.load('./Data_Square/f.npy')
    u = np.load('./Data_Square/u.npy')
    d = np.load('./Data_Square/d.npy')

    data_set = np.concatenate((x, y, f, u, d), axis=1)

    x = np.load('./Data_Square/x_b.npy')
    y = np.load('./Data_Square/y_b.npy')
    f = np.load('./Data_Square/f_b.npy')
    u = np.load('./Data_Square/u_b.npy')
    d = np.load('./Data_Square/d_b.npy')

    data_set_b = np.concatenate((x, y, f, u, d), axis=1)

    return np.concatenate((data_set, data_set_b), axis=0)


def load_2d_boun():
    x = np.load('./Data_Square/x_b.npy')
    y = np.load('./Data_Square/y_b.npy')
    f = np.load('./Data_Square/f_b.npy')
    u = np.load('./Data_Square/u_b.npy')
    d = np.load('./Data_Square/d_b.npy')

    data_set_b = np.concatenate((x, y, f, u, d), axis=1)

    return data_set_b
