
import torch
import torch.nn as nn
import numpy as np
import HW3
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
        for temperature in [1, 3, 20]:
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

    # Hyper-parameters
    embed_size = 220
    hidden_size = 220
    num_layers = 2
    num_epochs = 10
    num_samples = 30  # number of words to be sampled
    batch_size = 20
    seq_length = 30
    dropout = 0.3
    learning_rate = 0.005
    seed = 621
    date = datetime.datetime.now().strftime("%d-%m-%y %H-%M-%S")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    corpus = Corpus()
    ids = corpus.get_data('data/train.txt', batch_size)  # divide to batch size
    valid_d = corpus.get_data('data/valid.txt', batch_size)
    test_d = corpus.get_data('data/test.txt', batch_size)
    vocab_size = len(corpus.dictionary)
    num_batches = ids.size(1) // seq_length


    model = HW3.RNNLM(vocab_size, embed_size, hidden_size, num_layers, dropout)
    model.load_state_dict(torch.load(model_name,map_location=device))
    model.eval()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        model.cuda()
    else:
        model.cpu()
        # Set the random seed manually for reproducibility.

    generate()
    print(' ****** END of generative process ******')

