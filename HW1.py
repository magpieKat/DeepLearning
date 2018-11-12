'''
Deep Learning 097200
Nov 2018, ex.1
Submitted by:
Ariel Weigler 300564267
Liron McLey 200307791
Matan Hamra 300280310
'''

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class OurModel(nn.Module):
    def __init__(self):
        super(OurModel, self).__init__()
        self.linear1 = nn.Linear(28*28, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 128)
        self.linear4 = nn.Linear(128, output_size)
        self.ReLU = nn.ReLU()
        self.Softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.ReLU(self.linear1(x))
        x = self.ReLU(self.linear2(x))
        x = self.ReLU(self.linear3(x))
        output = self.Softmax(self.linear4(x))
        return output


def train():
    train_loss_vec = []
    train_error_vec = []
    test_loss_vec = []
    test_error_vec = []
    for epoch in range(num_epochs):
        running_train_loss, running_train_error = 0.0, 0.0
        running_test_loss, running_test_error = 0.0, 0.0
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
        loss, err = test(train_loader, 'train', epoch)
        running_train_loss += loss
        running_train_error += err

        train_loss_vec.append(round(running_train_loss, 4))
        train_error_vec.append(running_train_error)
        test_loss, test_err = test(test_loader, 'test', epoch)
        test_loss_vec.append(round(test_loss, 4))
        test_error_vec.append(test_err)
        print('--------------------------------------------')
    return train_loss_vec, train_error_vec, test_loss_vec, test_error_vec


def test(loader, dset, ep):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.reshape(-1, 28 * 28)
            outputs = model(images)
            loss = criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()


        print('Epoch: ', ep + 1, ', accuracy of the ', dset, 'set {} %'.format(100 * correct / total))
        return loss, 1.0 - (correct.item()/total)


def plot_graph(vec1, title1, vec2, title2, suptitle):
    plt.figure()
    plt.suptitle(suptitle, fontsize=25)
    plt.plot(vec1, color='blue', linewidth=2.5, linestyle='-', label=title1)
    plt.plot(vec2, color='red', linewidth=2.5, linestyle='-', label=title2)
    plt.legend(loc='upper left')


if __name__ == '__main__':
    # load and normalize the training & test data sets
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

    # hyper parameters
    output_size = 10
    num_epochs = 20
    batch_size = 100
    learning_rate = 0.05
    momentum = 0.9
    device = torch.device('cpu')

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    print('total training batch number: ', len(train_loader))
    print('total testing batch number: ', len(test_loader))

    model = OurModel().to(device)

    # calculate the number of trainable parameters in the model
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of trainable parameters: ', params)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # train the model
    train_lv, train_ev, test_lv, test_ev = train()
    plot_graph(train_lv, 'Train', test_lv, 'Test', 'Loss')
    plot_graph(train_ev, 'Train', test_ev, 'Test', 'Error')
    plt.show()
    # Save the model checkpoint
    torch.save(model.state_dict(), 'model2.pkl')