from Network import Network
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Created to plot convergence curves based on the accuracy and loss values obtained during training
def plot_figures(accuracies_train, losses_train, accuracies_validation, losses_validation):
    accuracies_train, losses_train = np.array(accuracies_train), np.array(losses_train)
    accuracies_validation, losses_validation = np.array(accuracies_validation), np.array(losses_validation)

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("training and validation results")
    ax1 = fig.add_subplot()
    ax1.set_xlabel('epoches')
    ax2 = ax1.twinx()

    ax1.plot(accuracies_train, label='train accuracy')
    ax1.plot(accuracies_validation, label='validation accuracy')

    ax2.plot(losses_train, color='r', label='train loss')
    ax2.plot(losses_validation, color='g', label='validation loss')

    ax1.set_ylabel("accuracy (%)")
    ax2.set_ylabel("loss")

    ax1.legend(loc='center left')
    ax2.legend(loc='center right')
    plt.xticks([])
    plt.show()


def main():

    network = Network()

    train_inputs = np.load('./data/train_inputs.npy')
    train_targets = np.load('./data/train_targets.npy').reshape(len(train_inputs), 1)
    print("-> Training data is loaded.")

    train_data = np.concatenate((train_inputs, train_targets), axis=1)
    np.random.shuffle(train_data)
    train_inputs, train_targets = train_data[:, 0:3], train_data[:, 3]
    print("-> Training data is shuffled.")


    validation_inputs = np.load('./data/valid_inputs.npy')
    validation_targets = np.load('./data/valid_targets.npy')
    print("-> Validation data is loaded.")

    batch_size = 100
    number_of_epoch = 20
    learning_rate = 0.01
    number_of_batch = int(train_inputs.shape[0]/batch_size)

    print("-> Batch size: ", batch_size, ", number of epoch: ", number_of_epoch, ", learning rate: ", learning_rate)

    losses_train, accuracies_train = [], []
    losses_validation, accuracies_validation = [], []

    losses_train_plot, accuracies_train_plot = [], []
    losses_validation_plot, accuracies_validation_plot = [], []
    counter = 0

    # We define a plot parameter to enable/disable  plotting.
    # For now: we initial to False to speed up training
    plot = False

    for epoch in range(number_of_epoch):
        print("-> Epoch: ", epoch)
        for i in range(number_of_batch):
            input1 = np.eye(250)[train_inputs[i*batch_size:(i+1)*batch_size, 0]]
            input2 = np.eye(250)[train_inputs[i*batch_size:(i+1)*batch_size, 1]]
            input3 = np.eye(250)[train_inputs[i*batch_size:(i+1)*batch_size, 2]]
            expected_outputs = np.eye(250)[train_targets[i*batch_size:(i+1)*batch_size]]

            loss_train, probabilities, X1, X2 = network.forward_propagation(input1, input2, input3, expected_outputs)

            d_w1, d_w2, d_w3, d_b1, d_b2 = network.backward_propagation(input1, input2, input3, probabilities, X1, X2, expected_outputs)

            accuracy_train = network.accuracy_calculation(expected_outputs, probabilities)

            network.update_parameters(d_w1, d_w2, d_w3, d_b1, d_b2, learning_rate)

            losses_train.append(loss_train)
            accuracies_train.append(accuracy_train)

            # We keep some intermediate accuracies and losses to plot convergence curves
            if plot == True and counter % 100 == 0 and counter > 0:
                losses_train_plot.append(np.sum(losses_train[-100:])/100)
                accuracies_train_plot.append(np.sum(accuracies_train[-100:])/100)

                loss_validation_, accuracy_validation_ = network.evaluation(validation_inputs, validation_targets,batch_size)
                losses_validation_plot.append(loss_validation_)
                accuracies_validation_plot.append(accuracy_validation_)
            counter += 1

        loss_validation, accuracy_validation = network.evaluation(validation_inputs, validation_targets, batch_size)
        losses_validation.append(loss_validation)
        accuracies_validation.append(accuracy_validation)

    loss_train, accuracy_train = network.evaluation(train_inputs, train_targets, batch_size)
    print("-> Final train accuracy is {:.3f}".format(accuracy_train), "%")

    loss_validation, accuracy_validation = network.evaluation(validation_inputs, validation_targets, batch_size)
    print("-> Final validation accuracy is {:.3f}".format(accuracy_validation), "%")

    pickle.dump(network, open('new_model.pk', 'wb'))

    print("-> new_model.pk is saved.")
    print("-> Execution completed.")

    if plot == True: plot_figures(accuracies_train_plot, losses_train_plot, accuracies_validation_plot, losses_validation_plot)

if __name__ == "__main__":
    main()