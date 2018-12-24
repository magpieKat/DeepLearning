'''
Deep Learning 097200
Dec 2018, ex.2
Submitted by:
Ariel Weigler 300564267
Liron McLey 200307791
Matan Hamra 300280310
'''

from datetime import date, datetime, time
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import datetime


class Block(nn.Module):
    def __init__(self, in_planes, out_planes, kernel, stride, padding, dropout_p, non_linearity):
        super(Block, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dropout_p = dropout_p
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_planes)

        if non_linearity == 'ReLU':
            self.NL = nn.ReLU()
        else:
            self.NL = nn.LeakyReLU()

        self.Dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        out = self.Dropout(self.NL(self.bn(self.conv(x))))
        return out


class Model(nn.Module):
    # (in_planes, out_planes, kernel, stride, padding, dropout_p, non_linearity)
    cfg = [(3,  32, 3, 1, 1, 0,   'ReLU'),
           (32, 68, 2, 1, 1, 0.1, 'ReLU')]

    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        self.layers = self.makeLayers()
        self.fc = nn.Linear(74052, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def makeLayers(self):
        layers = []
        for in_planes, out_planes, kernel, stride, padding, dropout_p, non_linearity in self.cfg:
            layers.append(Block(in_planes, out_planes, kernel, stride, padding, dropout_p, non_linearity))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return self.logsoftmax(out)

def to_cuda(x):
    if use_gpu:
        x = x.cuda()
    return x

def train():
    train_loss_vec = []
    train_error_vec = []
    test_loss_vec = []
    test_error_vec = []
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            # images = images.view(3,32,32)
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
            # images = images.reshape(-1, 3,32*32)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        accuracy = 100.0 * (correct.item() / total)
        print('Epoch: %d, accuracy of the %s set is: %.4f' % (ep+1, dat_set, accuracy))
        return loss/num_batches, 1.0 - (correct.item()/total)


def plot_graph(vec1, title1, vec2, title2, st, date):
    X = np.linspace(1, num_epochs, num_epochs)
    plt.figure()
    plt.suptitle(st, fontsize=25)
    plt.plot(X, vec1, color='blue', linewidth=2.5, linestyle='-', label=title1)
    plt.plot(X, vec2, color='red', linewidth=2.5, linestyle='-', label=title2)
    plt.xticks(np.arange(1, num_epochs, step=1))
    plt.legend(loc='upper right')
    plt.savefig('saveDir/'+date+st)


if __name__ == '__main__':
    # Load and normalize data, define hyper parameters:
    # Hyper Parameters
    num_epochs = 10
    batch_size = 100
    learning_rate = 0.001

    #  track time
    tic = time.time()

    # Image Preprocessing
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.2434, 0.2615)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.2434, 0.2615)),
    ])

    # CIFAR-10 Dataset
    train_dataset = dsets.CIFAR10(root='./data/',
                                  train=True,
                                  transform=transform_train,
                                  download=True)

    test_dataset = dsets.CIFAR10(root='./data/',
                                 train=False,
                                 transform=transform_test,
                                 download=True)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    use_gpu = torch.cuda.is_available()  # global bool
    device = torch.device('cpu')
    model = Model().to(device)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Calculate the number of trainable parameters in the model
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of trainable parameters: ', params)

    if params > 50000:
        print('# of parameters are greater than 50,000')
    num_batches = len(train_dataset) / batch_size

    date = datetime.datetime.now().strftime("%d-%m-%y %H-%M-%S")

    # Train the model
    train_lv, train_ev, test_lv, test_ev = train()
    plot_graph(train_lv, 'Train', test_lv, 'Test', 'Loss', date)
    plot_graph(train_ev, 'Train', test_ev, 'Test', 'Error', date)

    Accu = str(1.0 - round(test_ev[-1], 4))
    moduleName = 'saveDir/' + date + 'Accu_' + Accu + '_model.pkl'
    torch.save(model.state_dict(), moduleName)
    print('END')

    # track time
    toc = time.time()
    print('Elapsed: %s' % (toc - tic))

