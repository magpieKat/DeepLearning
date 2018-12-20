'''
Deep Learning 097200
Nov 2018, ex.1
Submitted by:
Ariel Weigler 300564267
Liron McLey 200307791
Matan Hamra 300280310
'''

from datetime import date, datetime

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
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


class OurModel(nn.Module):
    def __init__(self):
        super(OurModel, self).__init__()
        self.linear1 = nn.Linear(28*28, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 128)
        self.linear4 = nn.Linear(128, output_size)
        self.ReLU = nn.ReLU()
        self.LReLU = nn.LeakyReLU()
        self.Softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.LReLU(self.linear1(x))
        x = self.LReLU(self.linear2(x))
        x = self.ReLU(self.linear3(x))
        output = self.linear4(x)
        # output = self.Softmax(self.linear4(x))
        return output


def train():
    train_loss_vec = []
    train_error_vec = []
    test_loss_vec = []
    test_error_vec = []
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 28*28)
            model.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % batch_size == 0:
                print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                      % (epoch + 1, num_epochs, batch_idx + 1, len(train_dataset) // batch_size, loss.item()))
        loss_val, err_val = test(train_loader, 'train', epoch)
        train_loss_vec.append(round(loss_val, 4))
        train_error_vec.append(err_val)
        model.eval()
        test_loss, test_err = test(test_loader, 'test', epoch)
        test_loss_vec.append(round(test_loss, 4))
        test_error_vec.append(test_err)
        print('--------------------------------------------')
    return train_loss_vec, train_error_vec, test_loss_vec, test_error_vec


def test(loader, dat_set, ep):
    with torch.no_grad():
        loss = 0
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.reshape(-1, 28 * 28)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        accuracy = 100.0 * (correct.item() / total)
        print('Epoch: %d, accuracy of the %s set is: %.4f' % (ep+1, dat_set, accuracy))
        return loss/num_batches, 1.0 - (correct.item()/total)


def plot_graph(vec1, title1, vec2, title2, st, date,info):
    X = np.linspace(1, num_epochs, num_epochs)
    plt.figure()
    # full screen figure
    figManager = plt.get_current_fig_manager()
    figManager.full_screen_toggle()
    plt.suptitle(st+'\n'+'Momentum= %s Epoches= %d batch_size= %d learning_rate= %s' % info, fontsize=12)
    plt.plot(X, vec1, color='blue', linewidth=2.5, linestyle='-', label=title1)
    plt.plot(X, vec2, color='red', linewidth=2.5, linestyle='-', label=title2)
    plt.legend(loc='upper right')
    plt.savefig('saveDir/'+date+st)


if __name__ == '__main__':
    # Load and normalize the training & test data sets
    train_dataset = dsets.MNIST(root='./data',
                                train=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))]),
                                download=True)

    test_dataset = dsets.MNIST(root='./data',
                               train=False,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))]))

    # Hyper parameters
    output_size = 10
    num_epochs = 30
    batch_size = 50
    learning_rate = 0.05
    momentum = 0.22
    device = torch.device('cpu')
    info = (momentum, num_epochs, batch_size, learning_rate)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    print('total training batch number: ', len(train_loader))
    print('total testing batch number: ', len(test_loader))

    model = OurModel().to(device)

    # Calculate the number of trainable parameters in the model
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of trainable parameters: ', params)

    num_batches = len(train_dataset)/batch_size
    print('Number of batches: ', num_batches)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    date = datetime.datetime.now().strftime("%d-%m-%y %H-%M-%S")

    # Train the model
    train_lv, train_ev, test_lv, test_ev = train()
    plot_graph(train_lv, 'Train', test_lv, 'Test', 'Loss', date,info)
    plot_graph(train_ev, 'Train', test_ev, 'Test', 'Error', date,info)

    Accu = str(1.0 - round(test_ev[-1],4))
    moduleName = 'saveDir/'+date + 'Accu_' + Accu +'_model.pkl'
    torch.save(model.state_dict(), moduleName)
    print('END')