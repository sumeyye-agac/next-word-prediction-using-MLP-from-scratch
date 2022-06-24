import pickle
import numpy as np

print("-> Testing is started.")
test_inputs = np.load('./data/test_inputs.npy')
test_targets = np.load('./data/test_targets.npy')

network = pickle.load(open('model.pk','rb'))
print("--> model.pk is loaded.")

loss, accuracy = network.evaluation(test_inputs, test_targets, 1)

print("--> Test accuracy: {:.3f}".format(accuracy), "%")

print("-> End of the test.")
print("-"*50)

print("-> Next word prediction is started. Used model is model.pk.")

# Pick three words and predict the forth word using trained model
def predict_next_word(network, word1, word2, word3):
    vocabulary = np.load('./data/vocab.npy')
    try:
        one_hot_index1 = np.eye(250)[list(vocabulary).index(word1)]
        one_hot_index2 = np.eye(250)[list(vocabulary).index(word2)]
        one_hot_index3 = np.eye(250)[list(vocabulary).index(word3)]
        empty_expected_index = np.eye(250)[0]
    except:
        return "Word error. Please try with new words."

    _, probability, _, _ = network.forward_propagation([one_hot_index1], [one_hot_index2], [one_hot_index3], empty_expected_index)

    index_of_next_word = list(probability[0]).index(np.max(probability[0]))

    return vocabulary[index_of_next_word]

print("--> Prediction results:")
print('city of new ->', predict_next_word(network, 'city', 'of', 'new'))
print('life in the ->', predict_next_word(network, 'life', 'in', 'the'))
print('he is the ->', predict_next_word(network, 'he', 'is', 'the'))

print("--> Additional predictions:")

print('it is my ->', predict_next_word(network, 'it', 'is', 'my'))
print('i want to ->', predict_next_word(network, 'i', 'want', 'to'))
print('a part of ->', predict_next_word(network, 'a', 'part', 'of'))
print('it was much ->', predict_next_word(network, 'it', 'was', 'much'))
print('what other people ->', predict_next_word(network, 'what', 'other', 'people'))
print('four , three ->', predict_next_word(network, 'four', ',', 'three'))

print("-> End of the prediction.")