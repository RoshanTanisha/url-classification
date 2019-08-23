from sklearn.svm import SVC
from tflearn import DNN, lstm, dropout, regression, input_data, fully_connected
import tensorflow as tf

tf.reset_default_graph()


def get_SVM():
    return SVC()


def create_model(net):
    net = regression(net, optimizer='adam', loss='categorical_crossentropy', name='output')

    return DNN(net)


def get_rnn_layers(num_of_features):
    input_layer = input_data(shape=[None, 1, num_of_features])
    net = dropout(input_layer, keep_prob=0.5)
    net = lstm(net, n_units=100, activation='tanh', inner_activation='sigmoid', return_seq=True)
    net = lstm(net, n_units=100, activation='tanh', inner_activation='sigmoid')
    net = fully_connected(net, n_units=1, activation='sigmoid')

    return create_model(net)
