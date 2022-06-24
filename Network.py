import numpy as np

class Network(object):

    # Parameterized constructor of network
    # initializations are made based on Xavier technique
    # https://cs230.stanford.edu/section/4/
    def __init__(self):
        self.w1 = np.random.normal(0, 1/16, size=(250, 16))
        self.w2 = np.random.normal(0, 1/128, size=(48, 128))
        self.w3 = np.random.normal(0, 1/250, size=(128, 250))
        self.b1 = np.zeros(128)
        self.b2 = np.zeros(250)

    # Cross entropy loss function
    # a small number (1e-20) is added to log in order to avoid:
    # "RuntimeWarning: divide by zero encountered in log"
    def cross_entropy_loss(self, actual, predicted):
        loss = -np.sum(actual * np.log(predicted + 1e-20)) / len(actual)
        return loss

    # Softmax function
    def softmax(self, x):
        # Numerical stability issue is handled.
        # We got "RuntimeWarning: overflow encountered in exp" before taking into account overflows.
        x = x - x.max(1).reshape((-1, 1))
        e = np.exp(x)
        return e/e.sum(1).reshape((-1, 1))

    # Sigmoid activation function
    def sigmoid(self, x):
        y = 1. / (1. + np.exp(-x))
        return y

    # Forward propagation steps
    def forward_propagation(self, input1, input2, input3, expected_outputs):
        e1 = np.dot(input1, self.w1)                # embedding of first word (nx16)
        e2 = np.dot(input2, self.w1)                # embedding of second word (nx16)
        e3 = np.dot(input3, self.w1)                # embedding of third word (nx16)
        e1_e2 = np.concatenate((e1, e2), axis=1)
        X1 = np.concatenate((e1_e2, e3), axis=1)    # embedding of three words (nx48)
        h = np.add(np.dot(X1, self.w2), self.b1)    # hidden layer (nx128)
        X2 = self.sigmoid(h)                        # output of sigmoid (nx128)
        o = np.add(np.dot(X2, self.w3), self.b2)    # output layer (nx250)
        probabilities = self.softmax(o)             # probability values (nx250)
        loss = self.cross_entropy_loss(expected_outputs, probabilities) # a single value

        return loss, probabilities, X1, X2

    # Backward propagations steps
    def backward_propagation(self, input1, input2, input3, probabilities, X1, X2, expected_outputs):
        derror = probabilities - expected_outputs      # error 1: from loss to output layer (nx250)
        w3_gradient = np.dot(X2.transpose(), derror)   # gradients of weight 3 (128x250)
        b2_gradient = np.sum(derror, axis=0)           # gradients of bias 2 (250)
        derror1 = np.dot(derror, self.w3.transpose()) * X2 * (1 - X2)  # error 2: from loss to hidden layer (nx128)
        w2_gradient = np.dot(X1.transpose(), derror1)  # gradients of weight 2 (48x128)
        b1_gradient = np.sum(derror1, axis=0)          # gradients of bias 1 (128)
        derror2 = np.dot(derror1, self.w2.transpose()) # error 3: from loss to X1 (nx48)
        inputs = [input1, input2, input3]
        w1_gradient = np.zeros((250, 16))              # initialization for gradients of weight 1 (250x16)
        # Inverse of concatanation while calculating w1_gradients: Split into three
        for word in range(0, 3):
            w1_gradient = w1_gradient + np.dot(inputs[word].transpose(), derror2[:, (word) * 16:(word + 1) * 16])

        return w1_gradient, w2_gradient, w3_gradient, b1_gradient, b2_gradient

    # Accuracy calculation
    # For each data point, among 250 word probabilities in its predicted list,
    # we took the index of the value which has the highest probability as predicted_index
    # and then compare it with expected_index to calculate accuracy
    # accuracy result is calculated in %
    def accuracy_calculation(self, expected_outputs, predicted):
        true_predicted = 0
        for i in range(len(expected_outputs)):
            expected_index = list(expected_outputs[i]).index(1.)
            predicted_index = list(predicted[i]).index(np.max(predicted[i]))
            if expected_index == predicted_index: true_predicted += 1
        accuracy = 100. * true_predicted / (len(expected_outputs))
        return accuracy

    # Update weights and biases by using gradients and learning rate
    def update_parameters(self, d_w1, d_w2, d_w3, d_b1, d_b2, learning_rate):
        self.w1 = self.w1 - learning_rate * d_w1
        self.w2 = self.w2 - learning_rate * d_w2
        self.w3 = self.w3 - learning_rate * d_w3
        self.b1 = self.b1 - learning_rate * d_b1
        self.b2 = self.b2 - learning_rate * d_b2

    # Evaluate the network on a another data
    # During trainig, we give validation data with batch_size that we use in training
    # In the test phase, we call this function with a batch_size = 1 (one single point at a time)
    def evaluation(self, inputs, targets, batch_size):
        tot_loss, tot_accuracy = 0., 0.
        number_of_batch = int(inputs.shape[0] / batch_size)

        for i in range(number_of_batch):
            input1 = np.eye(250)[inputs[i * batch_size:(i + 1) * batch_size, 0]]  # (nx250)
            input2 = np.eye(250)[inputs[i * batch_size:(i + 1) * batch_size, 1]]  # (nx250)
            input3 = np.eye(250)[inputs[i * batch_size:(i + 1) * batch_size, 2]]  # (nx250)
            expected_outputs = np.eye(250)[targets[i * batch_size:(i + 1) * batch_size]]  # (nx250)

            loss, probabilities, _, _ = self.forward_propagation(input1, input2, input3, expected_outputs)
            accuracy = self.accuracy_calculation(expected_outputs, probabilities)
            tot_accuracy += accuracy
            tot_loss += loss

        return tot_loss/number_of_batch, tot_accuracy/number_of_batch
