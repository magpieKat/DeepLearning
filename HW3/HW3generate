
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
from data_utils import Dictionary, Corpus
import matplotlib.pyplot as plt
import datetime
import math
import os

def generate():
    # Test the model
    torch.no_grad()
    file_name = 'saveDir/'+date+'WordCompletion.txt'
    with open(file_name, 'w') as outf:
        for temperature in [1, 10, 100]:
            # Set initial hidden and cell states
            state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                     torch.zeros(num_layers, 1, hidden_size).to(device))
            outf.write('Output for temperature ' + np.str(temperature) + ' is:\n')

            for sentenceWord in ['buy', 'low', 'sell', 'high', 'is']:
                input = torch.LongTensor([[corpus.dictionary.word2idx[sentenceWord]], ]).to(device)
                output, state = model(input, state)

            outf.write('buy low sell high is the ')
            input = torch.LongTensor([[corpus.dictionary.word2idx['the']], ]).to(device)

            for i in range(num_samples):
                # Forward propagate RNN
                output, state = model(input, state)

                # Sample a word id
                prob = output.squeeze().div(temperature).exp()
                word_id = torch.multinomial(prob, num_samples=1).item()

                # Fill input with sampled word id for the next time step
                input.fill_(word_id)

                # File write
                word = corpus.dictionary.idx2word[word_id]
                word = '\n' if word == '<eos>' else word + ' '
                outf.write(word)

                if (i + 1) % 100 == 0:
                    print('Sampled [{}/{}] words and save to {}'.format(i + 1, num_samples, 'sample.txt'))

            outf.write('\n\n')


if __name__ == '__main__':

    model_name='saveDir/model.pkl'
    datapath='./data'
    seed=621

    with open(model_name, 'rb') as f:
        model = torch.load(f)
    model.eval()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        model.cuda()
    else:
        model.cpu()

    generate()
    print(' ****** END of generative process ******')

