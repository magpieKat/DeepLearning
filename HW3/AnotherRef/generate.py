###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import torch
from torch.autograd import Variable
import numpy as np
import data

# Model parameters.
checkpoint='./model.pkl'
cuda=True
datapath='./data'
log_interval=100
outf='generated.txt'
seed=1111 #'Buy low, sell high is the'
temperature=100
words=30

# Set the random seed manually for reproducibility.
torch.manual_seed(seed)
if torch.cuda.is_available():
    if not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(seed)

with open(checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()

if cuda:
    model.cuda()
else:
    model.cpu()

corpus = data.Corpus(datapath)
ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)
input = Variable(torch.LongTensor([[corpus.dictionary.word2idx['buy']],]), volatile=True)
if cuda:
    input.data = input.data.cuda()
    
with open(outf, 'w') as outf:
    output, hidden = model(input, hidden)
    input = Variable(torch.LongTensor([[corpus.dictionary.word2idx['low']],]), volatile=True)
    if cuda:
        input.data = input.data.cuda()
    output, hidden = model(input, hidden)
    input = Variable(torch.LongTensor([[corpus.dictionary.word2idx['sell']],]), volatile=True)
    if cuda:
        input.data = input.data.cuda()
    output, hidden = model(input, hidden)
    input = Variable(torch.LongTensor([[corpus.dictionary.word2idx['high']],]), volatile=True)
    if cuda:
        input.data = input.data.cuda()
    output, hidden = model(input, hidden)
    input = Variable(torch.LongTensor([[corpus.dictionary.word2idx['is']],]), volatile=True)
    if cuda:
        input.data = input.data.cuda()
    output, hidden = model(input, hidden)
    input = Variable(torch.LongTensor([[corpus.dictionary.word2idx['the']],]), volatile=True)
    if cuda:
        input.data = input.data.cuda()
    output, hidden = model(input, hidden)
    outf.write('Output for temperature - ' + np.str(temperature) + ' is:\n')
    outf.write('Buy low, sell high is the... \n')
    for i in range(words):
        output, hidden = model(input, hidden)
        word_weights = output.squeeze().data.div(temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        input.data.fill_(word_idx)
        word = corpus.dictionary.idx2word[word_idx]

        outf.write(word + ('\n' if i % 20 == 19 else ' '))

        if i % log_interval == 0:
            print('| Generated {}/{} words'.format(i, words))
