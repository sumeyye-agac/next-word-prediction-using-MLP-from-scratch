# Sumeyye Agac - 2018800039
# CmpE 597 - Sp. Tp. (Spring 2021)
# Project #1
# tsne.py

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

vocab = np.load('./data/vocab.npy')
print("-> vocab.npy is loaded.")
one_hot_matrix = np.identity(250)

network = pickle.load(open('model.pk','rb'))
print("-> model.pk is loaded.")

embeddings = np.dot(one_hot_matrix, network.w1) # (250x16)
embeddings_2d = TSNE(n_components=2).fit_transform(embeddings) # (250x2)
print("-> 2d embeddings are created.")
np.set_printoptions(suppress=True)

x_coords, y_coords = embeddings_2d[:, 0], embeddings_2d[:, 1]

plt.scatter(x_coords, y_coords, s=0)

for label, x, y in zip(vocab, x_coords, y_coords):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

plt.savefig('tsne.png')
print("-> tsne.png saved.")
plt.show()
