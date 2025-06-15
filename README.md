# Haiku Generator

This project uses neural network built with pytorch to generate haikus.
It includes various methodologies to generate haiku including using embeddings, RNN and advanced methods in the future.
This is trained on a dataset of haikus provided in haikus.txt.

The initial model is one layer neural network with one hot encoded tensor as input and the output is sampled
by the torch.multinomial function from the probability distribution predicted.

The embeddings model is three layered neural network with embeddings of size 20 in the first layer, following
the paper "A neural probabilistic language model" by Bengio et al and the output is similarly sampled.

## How to run
- Make sure to install pytorch
- Run  ' python "required model's file name".py'

## Author
Vijay Varadarajan
