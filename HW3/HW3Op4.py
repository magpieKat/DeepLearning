import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
from data_utils import Dictionary, Corpus
import matplotlib.pyplot as plt


# RNN based language model
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.33)
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
    # Train the model
    for epoch in range(num_epochs):
        model.train()
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

            # Backward and optimize
            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()

            step = (i + 1) // seq_length
            if step % 100 == 0:
                print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                      .format(epoch + 1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))
        validiate(epoch,states)
        test(epoch,states)
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
        test_loss =   test_loss / (test_d.size(1) // seq_length)
        print('\n| end of epoch {:3d} | test loss {:5.2f} | '
              'test ppl {:8.2f}\n'.format(epoch+1, test_loss, np.exp(test_loss)))


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
    embed_size = 128
    hidden_size = 128
    num_layers = 2
    num_epochs = 20
    num_samples = 30 # number of words to be sampled
    batch_size = 50
    seq_length = 30
    learning_rate = 0.0025

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
    print(vocab_size)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Save the model checkpoints

    train()  # train the model



    generate()  # test the model
    # generate()        # generate
    print('*****END*****')
