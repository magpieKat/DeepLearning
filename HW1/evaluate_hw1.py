

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import HW1 as hw1


def test(loader):
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.reshape(-1, 28 * 28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        accuracy = 100.0 * (correct.item() / total)
        return accuracy


if __name__ == '__main__':
    # load MNIST test set
    # Hyper parameters
    batch_size = 60
    output_size = 10
    device = torch.device('cpu')

    test_dataset = dsets.MNIST(root='./data',
                               train=False,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))]))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    # load trained network
    model = hw1.OurModel()
    model.load_state_dict(torch.load('21-11-18 17-47-28Accu_0.9822_model.pkl'))

    # return average accuracy over test set
    num_batches = int(len(test_dataset) / batch_size)
    acc_vec = []
    # for i in range(num_batches):
    test_err = test(test_loader)
        # acc_vec.append(test_err)

    print('Test Error: %.4f' % test_err)
    print('END')
