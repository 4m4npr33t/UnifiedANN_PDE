import tensorflow as tf


class neural_network:
    def __init__(self, n_input, n_output, n_hidden_units, weight_initialization=tf.contrib.layers.xavier_initializer(),
                 activation_hidden=tf.nn.sigmoid, name='model_'):
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden_units = n_hidden_units
        self.weight_initialization = weight_initialization
        self.activation_hidden = activation_hidden

        self.weights = {}
        self.biases = {}
        self.number_of_layers = len(self.n_hidden_units)

        self.name = name

        for i in range(0, self.number_of_layers):
            if i == 0:
                self.weights[self.name + '0'] = tf.get_variable(self.name + 'weight_' + str(0),
                                                                shape=[self.n_input, self.n_hidden_units[0]],
                                                                initializer=self.weight_initialization,
                                                                dtype=tf.float64)
            else:
                self.weights[self.name + str(i)] = tf.get_variable(self.name + 'weight_' + str(i),
                                                                   shape=[self.n_hidden_units[i - 1],
                                                                          self.n_hidden_units[i]],
                                                                   initializer=self.weight_initialization,
                                                                   dtype=tf.float64)

            self.biases[self.name + str(i)] = tf.get_variable(self.name + 'bias_' + str(i),
                                                              shape=[self.n_hidden_units[i]],
                                                              initializer=self.weight_initialization, dtype=tf.float64)

        self.weights[self.name + str(self.number_of_layers)] = tf.get_variable(
            self.name + 'weight_' + str(self.number_of_layers), shape=[self.n_hidden_units[-1], self.n_output],
            initializer=self.weight_initialization, dtype=tf.float64)
        self.biases[self.name + str(self.number_of_layers)] = tf.get_variable(
            self.name + 'bias_' + str(self.number_of_layers), shape=[self.n_output],
            initializer=self.weight_initialization, dtype=tf.float64)


    def value(self, input_var):
        for i in range(0, self.number_of_layers):
            if i == 0:
                layer = tf.add(tf.matmul(input_var, self.weights[self.name + '0']), self.biases[self.name + '0'])
            else:
                layer = tf.add(tf.matmul(layer, self.weights[self.name + str(i)]), self.biases[self.name + str(i)])

            layer = self.activation_hidden(layer)

        return tf.matmul(layer, self.weights[self.name+str(self.number_of_layers)]) + self.biases[self.name+str(self.number_of_layers)]

    def dx(self, X):
        dx = tf.gradients(self.value(X),X)
        return dx[0]

    def d2x(self, X):
        d2x = tf.gradients(self.dx(X),X)
        return d2x[0]

    def compute_lu(self, X, d2g, d, d2d, d1d, ret):
        d1_x = self.dx(X)
        d2_x = self.d2x(X)
        x_val = self.value(X)
        for it in range(d2g.shape[1]):
            ret += tf.expand_dims(d2g[:, it], -1) + tf.multiply(tf.expand_dims(d2d[:, it], -1), x_val) + (2 * tf.multiply(tf.expand_dims(d1d[:, it], -1), tf.expand_dims(d1_x[:, it], -1))) + tf.multiply(d, tf.expand_dims(d2_x[:, it], -1))

        return ret