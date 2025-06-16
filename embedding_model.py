from string import punctuation
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import random

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

# building the inputs according to the size of context
context = 3
xs, ys = [], []
for i in range(len(haiku_list) - context):
    c = haiku_list[i:context+i]  # context block
    block = []
    for w in c:
        x = ''.join(chr for chr in w if chr not in removed_punctuation)
        block.append(haiku_to_idx[x])
    xs.append(block)
    y = ''.join(chr for chr in haiku_list[context + i] if chr not in removed_punctuation)
    ys.append(haiku_to_idx[y])

# shuffling the inputs and outputs

random.seed(42)
combined = list(zip(xs, ys))
random.shuffle(combined)
x_shuffled, y_shuffled = zip(*combined)

X = torch.tensor(x_shuffled)
Y = torch.tensor(y_shuffled)
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((len(unique_words), 20), generator = g)  # embeddings of size 10
W1 = torch.randn((60, 100), generator = g)
b1 = torch.randn(100, generator = g)
W2 = torch.randn((100, len(unique_words)), generator = g)
b2 = torch.rand(len(unique_words), generator = g)
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True
sum(p.nelement() for p in parameters)  # number of elements in the neural network

# training, validation and test sets
n1 = int(0.8 * len(x_shuffled))
n2 = int(0.9 * len(x_shuffled))

Xtr, Ytr = X[:n1], Y[:n1]
Xdev, Ydev = X[n1:n2], Y[n1:n2]
Xte, Yte = X[n2:], Y[n2:]

step, lossi = [], []
for _ in range(15000):

    # minibatch selection
    ix = torch.randint(0, len(Xtr), (32,))
    # forward pass
    emb = C[Xtr[ix]]  # embedding
    h = torch.tanh(emb.view(-1, 60) @ W1 + b1)  # tanh activated first layer
    logits = h @ W2 + b2  # second layer
    loss = -F.cross_entropy(logits, Ytr[ix]) + wd *((W1**2).mean() + (W2**2).mean() + (C**2).mean()) # log likelihood and regularization of weights and embeddings
    step.append(_)
    lossi.append(loss.item())
    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if _ < 10000 else 0.01
    for p in parameters:
        p.data += lr * p.grad  # gradient ascent

plt.plot(step, [-i for i in lossi])  # plotting the loss function(inverting it since it is negative)

# total training set
with torch.no_grad():
    emb = C[Xtr]
    h = torch.tanh(emb.view(-1, 60) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr)
    print("the training loss is", loss.item())

# validation
with torch.no_grad():
    emb = C[Xdev]
    h = torch.tanh(emb.view(-1, 60) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ydev)
    print("the validation loss is", loss.item())

# sampling from the model

context = xs[998]  # ghost context for initial guess

with torch.no_grad():
    for _ in range(20):
        emb = C[torch.tensor(context)]
        h = torch.tanh(emb.view(-1, 60) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim = 1)
        idx = torch.multinomial(probs, num_samples = 1, replacement = True, generator = g).item()
        context = context[1:] + [idx]
        print(idx_to_haiku[idx])