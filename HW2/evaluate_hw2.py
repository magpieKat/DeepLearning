import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import HW2_Option4 as hw2


def test(loader):
        model.eval()
        correct = 0
        total = 0
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        accuracy = 100.0 * (correct.item() / total)
        return accuracy


if __name__ == '__main__':
    # load CIFAR10 test set
    # Hyper parameters
    batch_size = 64
    output_size = 10
    device = torch.device('cuda')

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.2434, 0.2615)),
    ])
    test_dataset = dsets.CIFAR10(root='./data/',
                                 train=False,
                                 transform=transform_test,
                                 download=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              pin_memory=(torch.cuda.is_available()),
                                              num_workers=0,
                                              shuffle=False)

    # load trained network
    model = hw2.DenseNet()
    model.load_state_dict(torch.load('26-12-18 09-28-03Accu_0.9065_model.pkl'))

    # return average accuracy over test set
    num_batches = int(len(test_dataset) / batch_size)
    acc_vec = []
    # for i in range(num_batches):
    test_err = test(test_loader)
        # acc_vec.append(test_err)

    print('Test Error: %.4f' % test_err)
    print('END')
