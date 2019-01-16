import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
from data_utils import Dictionary, Corpus
import matplotlib.pyplot as plt
import datetime
import os


def new_dir(path, name):  # create new sub folder named: path/name
    newPath = path + '/' + name
    try:
        if not os.path.exists(newPath):
            os.makedirs(newPath)
    except OSError:
        print('Error: Creating directory of ' + newPath)
    return newPath


# RNN based language model
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.35)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)

        # Forward propagate LSTM
        out, (h, c) = self.lstm(x, h)
        out = self.dropout(out)
        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))

        # Decode hidden states of all time steps
        out = self.linear(out)
        return out, (h, c)

# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states]


def train():
    train_loss_vec = []
    test_loss_vec = []
    # Train the model

    for epoch in range(num_epochs):
        model.train()
        train_loss_val = 0

        # Set initial hidden and cell states
        states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                  torch.zeros(num_layers, batch_size, hidden_size).to(device))

        for i in range(0, ids.size(1) - seq_length, seq_length):
            # Get mini-batch inputs and targets
            inputs = ids[:, i:i + seq_length].to(device)
            targets = ids[:, (i + 1):(i + 1) + seq_length].to(device)

            # Forward pass
            states = detach(states)
            outputs, states = model(inputs, states)
            loss = criterion(outputs, targets.reshape(-1))
            train_loss_val = train_loss_val + loss / (ids.size(1) // seq_length)

            # Backward and optimize
            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            step = (i + 1) // seq_length
            if step % 100 == 0:
                print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                      .format(epoch + 1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))

        train_loss_vec.append(train_loss_val)
        validiate(epoch, states)
        test_loss_val = test(epoch, states)
        test_loss_vec.append(test_loss_val)
    return train_loss_vec, test_loss_vec


def validiate(epoch, states):
    global best_val_loss, learning_rate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i in range(0, valid_d.size(1) - seq_length, seq_length):
            inputs = valid_d[:, i:i + seq_length].to(device)
            targets = valid_d[:, (i + 1):(i + 1) + seq_length].to(device)

            outputs, states = model(inputs, states)
            crt = criterion(outputs, targets.reshape(-1))
            val_loss += crt
        val_loss = val_loss/(valid_d.size(1) // seq_length)
        print('\n| end of epoch {:3d} | valid loss {:5.2f} | '
              'valid ppl {:8.2f}\n'.format(epoch+1, val_loss, np.exp(val_loss)))
        if not best_val_loss or val_loss < best_val_loss:
            torch.save(model.state_dict(), 'model_Option4.pkl')
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            learning_rate /= 4.0


def plot_graph(vec1, title1, vec2, title2, st, date):
    X = np.linspace(1, num_epochs, num_epochs)
    plt.figure()
    plt.suptitle(st, fontsize=25)
    plt.plot(X, vec1, color='blue', linewidth=2.5, linestyle='-', label=title1)
    plt.plot(X, vec2, color='red', linewidth=2.5, linestyle='-', label=title2)
    plt.xticks(np.arange(1, num_epochs, step=1))
    plt.legend(loc='upper right')
    new_dir(os.getcwd(), 'saveDir')
    plt.savefig('saveDir/'+date+st)


def test(epoch, states):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i in range(0, test_d.size(1) - seq_length, seq_length):
            inputs = test_d[:, i:i + seq_length].to(device)
            targets = test_d[:, (i + 1):(i + 1) + seq_length].to(device)

            outputs, states = model(inputs, states)
            crt = criterion(outputs, targets.reshape(-1))
            test_loss += crt
        test_loss = test_loss / (test_d.size(1) // seq_length)
        print('\n| end of epoch {:3d} | test loss {:5.2f} | '
              'test ppl {:8.2f}\n'.format(epoch+1, test_loss, np.exp(test_loss)))

    return test_loss


def generate():
    # Test the model
    with torch.no_grad():
        with open('sample.txt', 'w') as f:
            # Set intial hidden ane cell states
            state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                     torch.zeros(num_layers, 1, hidden_size).to(device))

            # Select one word id randomly
            prob = torch.ones(vocab_size)
            input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)

            for i in range(num_samples):
                # Forward propagate RNN
                output, state = model(input, state)

                # Sample a word id
                prob = output.exp()
                word_id = torch.multinomial(prob, num_samples=1).item()

                # Fill input with sampled word id for the next time step
                input.fill_(word_id)

                # File write
                word = corpus.dictionary.idx2word[word_id]
                word = '\n' if word == '<eos>' else word + ' '
                f.write(word)

                if (i + 1) % 100 == 0:
                    print('Sampled [{}/{}] words and save to {}'.format(i + 1, num_samples, 'sample.txt'))


if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    embed_size = 77
    hidden_size = 202
    num_layers = 2
    num_epochs = 12
    num_samples = 5  # number of words to be sampled
    batch_size = 50
    seq_length = 30
    learning_rate = 0.0021

    # Load "Penn Treebank" dataset
    corpus = Corpus()
    ids = corpus.get_data('data/train.txt', batch_size)  # divide to batch size
    valid_d = corpus.get_data('data/valid.txt', batch_size)
    test_d = corpus.get_data('data/test.txt', batch_size)
    vocab_size = len(corpus.dictionary)
    num_batches = ids.size(1) // seq_length
    best_val_loss = None

    model = RNNLM(vocab_size, embed_size, hidden_size, num_layers).to(device)

    # Calculate the number of trainable parameters in the model
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of trainable parameters: ', params)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    # Save the model checkpoints

    train_lv, test_lv = train()  # train the model

    date = datetime.datetime.now().strftime("%d-%m-%y %H-%M-%S")
    plot_graph(train_lv, 'Train', test_lv, 'Test', 'Loss', date)
    plot_graph(np.exp(train_lv), 'Train', np.exp(test_lv), 'Test', 'Perplexity', date)

    generate()  # test the model
    # generate()        # generate
    print('*****END*****')
