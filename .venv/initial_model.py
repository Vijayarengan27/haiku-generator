# initializing the required packages

from string import punctuation
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from math import log
import torch.nn.functional as F

# creation of the training data

with open('dataset/haikus.txt') as f:  # loading the training data
    haiku_text = f.read().split()

haiku_list = list(haiku_text)
allowed_punctuation = ["'"]
removed_punctuation = set(list(punctuation)) - set(allowed_punctuation)  # create a list of punctuations to be removed
clean_haiku_list = []
for w in haiku_list:
    x = ''.join(chr for chr in w if chr not in removed_punctuation)
    clean_haiku_list.append(x)
unique_words = sorted(set(clean_haiku_list))

# lookup tables for building the torch tensor

haiku_to_idx = {word:index for index,word in enumerate(unique_words)}
idx_to_haiku = {value:key for key,value in haiku_to_idx.items()}

P = torch.zeros((len(unique_words), len(unique_words)), dtype = torch.float32)

for w1, w2 in zip(haiku_list, haiku_list[1:]):
    x = "".join(chr for chr in w1 if chr not in removed_punctuation)
    y = "".join(chr for chr in w2 if chr not in removed_punctuation)
    x1 = haiku_to_idx[x]
    y1 = haiku_to_idx[y]
    P[x1, y1] += 1

# visualization of the first fifty unique words and their frequencies of following words
plt.imshow(P[:50,:50])

g = torch.Generator().manual_seed(2147483647)
P /= P.sum(1, keepdim = True)  # getting the probabilities


# neural network

# initialization

W = torch.randn((1424, 1424), generator = g, requires_grad= True)
xs, ys = [], []
for w1, w2 in zip(haiku_list, haiku_list[1:]):
    x = ''.join(chr for chr in w1 if chr not in removed_punctuation)
    y = ''.join(chr for chr in w2 if chr not in removed_punctuation)
    xs.append(haiku_to_idx[x])
    ys.append(haiku_to_idx[y])


X = F.one_hot(torch.tensor(xs), num_classes = 1424).float()
size = X.shape
for _ in range(1000):
    # forward pass
    logits = X @ W
    counts = logits.exp()
    probs = counts/ counts.sum(1, keepdim = True)
    loss = -probs[np.arange(size[0]),ys].log().mean()

    #backward pass
    W.grad = None
    loss.backward()

    # update

    W.data += -25 * W.grad  # need a higher learning rate

# predict from the model and check with the normal bigram prediction

idx = 143  # initial ghost word given to write the next words in the poem

for _ in range(20):
    i = F.one_hot(torch.tensor([idx]), num_classes = 1424).float()
    logits = i @ W
    counts = logits.exp()
    probs = counts/counts.sum(1, keepdim = True)
    idx = torch.multinomial(probs, num_samples = 1, replacement = True, generator = g).item()
    print(idx_to_haiku[idx])
