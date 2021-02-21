import sys
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


class MyDataSet(data.Dataset):
    def __init__(self, examples, labels):
        self.examples = examples
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.examples[idx], self.labels[idx].type(torch.LongTensor)


class MyDataSetTest(data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class Model1(torch.nn.Module):
    """
    First model + Second model - Neural Network with two hidden layers, the first layer
    have a size of 100 and the second layer  have a size of 50, both
    followed by ReLU activation function and softmax activation function
    on the last layer.
    """
    def __init__(self,image_size):
        super(Model1, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1)


class Model3(torch.nn.Module):
    """
    Third model - Neural Network with two hidden layers, the first layer
    have a size of 100 and the second layer  have a size of 50, both
    followed by ReLU and dropout activation functions and softmax activation function
    on the last layer.
    """
    def __init__(self,image_size):
        super(Model3, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.dropout(x, p = 0.4)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p = 0.4)
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1)


class Model4(torch.nn.Module):
    """
    Fourth model - Neural Network with two hidden layers, the first layer
    have a size of 100 and the second layer  have a size of 50, both
    followed by batchNorm and then ReLU activation function.
    """
    def __init__(self,image_size):
        super(Model4, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.batchNorm1 = nn.BatchNorm1d(100)
        self.fc1 = nn.Linear(100, 50)
        self.batchNorm2 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.fc0(x)
        x = self.batchNorm1(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = self.batchNorm2(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1)


class Model5(torch.nn.Module):
    """
    Fifth model - Neural Network with two hidden layers, the first layer
    have a size of 128, the second layer have a size of 64, the third layer have a size of 10,
    the fourth layer have a size of 10, the fifth layer have a size of 10 and the sixth layer
    have a size of 10. All followed by ReLU activation function.
    """
    def __init__(self,image_size):
        super(Model5, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim = 1)


class Model6(torch.nn.Module):
    """
    Sixth model - Neural Network with two hidden layers, the first layer
    have a size of 128, the second layer have a size of 64, the third layer have a size of 10,
    the fourth layer have a size of 10, the fifth layer have a size of 10 and the sixth layer
    have a size of 10. All followed by sigmoid activation function.
    """
    def __init__(self,image_size):
        super(Model6, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.sigmoid(self.fc0(x))
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim = 1)

def train(model, train_loader, optimizer):
    model.train()
    correct=0
    test_loss=0
    for data, labels in train_loader:
        optimizer.zero_grad()
        output = model(data)
        test_loss += F.nll_loss(output, labels, size_average=False).item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).cpu().sum()
        loss = F.nll_loss(output,labels)
        loss.backward()
        optimizer.step()
    len = train_loader.__len__() *32
    test_loss /= len
    #print(str(test_loss))
    #print(str(100.* correct.item() /len))


def test(model, train_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output,target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).cpu().sum()
    len = train_loader.__len__() *32
    test_loss /= len
    #print(test_loss)
    #print(str(100.* correct.item() /len))


def create_test_y_file(model, test_loader):
    pred_f = open("test_y", "w")
    with torch.no_grad():
        for images in test_loader:
            # forward pass
            output = model(images)
            # get prediction with softmax activation function
            pred = output.max(1, keepdim=True)[1]
            # write prediction to test_y file
            for p in pred:
                pred_f.write(str(p.item()) +"\n")


if __name__ == '__main__':
    """ 
    # load data from torchvision
    transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    data_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms)
    data_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms)
    data_train_loader3 = DataLoader(data_train, batch_size=32, shuffle=True)
    data_test_loader3 = DataLoader(data_test, batch_size=32, shuffle=True)
    """
    # load data from files
    data_train_x = np.genfromtxt(sys.argv[1], delimiter=" ", dtype=np.float)
    data_train_y = np.genfromtxt(sys.argv[2], delimiter=' ', dtype=np.int)
    data_test_x = np.genfromtxt(sys.argv[3], delimiter=' ', dtype=np.float)
    # normalize data
    data_train_x = data_train_x/255
    data_test_x = data_test_x/255
    # convert data from numpy to torch
    data_train_x = torch.from_numpy(data_train_x)
    data_train_y = torch.from_numpy(data_train_y)
    data_test_x = torch.from_numpy(data_test_x)
    data_train_x = torch.as_tensor(data_train_x).float()
    data_test_x = torch.as_tensor(data_test_x).float()
    # create dataset
    transformed_dataset = MyDataSet(data_train_x, data_train_y)
    transformed_dataset_test = MyDataSetTest(data_test_x)

    train_size = int(0.8 * transformed_dataset.__len__())
    test_size = transformed_dataset.__len__() - train_size
    # create train set and validation set
    train_dataset, test_dataset = torch.utils.data.random_split(transformed_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    all_train_loader = DataLoader(transformed_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(transformed_dataset_test, batch_size=32, shuffle=False)

    # create the best model
    model4 = Model4(image_size=28 * 28)
    optimizer_SGD = optim.SGD(model4.parameters(), lr=0.09)

    """ 
    # all models:
    model1 = Model1(image_size=28 * 28)
    optimizer_SGD = optim.SGD(model1.parameters(), lr=0.09)
    model2 = Model1(image_size=28 * 28)
    optimizer_ADAM = optim.Adam(model2.parameters(), lr=0.02)
    model3 = Model3(image_size=28 * 28)
    optimizer_SGD = optim.SGD(model3.parameters(), lr=0.003)
    model4 = Model4(image_size=28 * 28)
    optimizer_SGD = optim.SGD(model4.parameters(), lr=0.09)
    model5 = Model5(image_size=28 * 28)
    optimizer_ADAM = optim.Adam(model5.parameters(), lr=0.01)
    model6 = Model6(image_size=28 * 28)
    optimizer_SGD = optim.Adam(model6.parameters(), lr=0.02)
    """
    for epoch in range(1, 11):
        # train and test data from torchvision
        #train(model4, data_train_loader3, optimizer_SGD)
        #test(model4, data_test_loader3)

        # train and test data from files
        train(model4, train_loader, optimizer_SGD)
        test(model4, val_loader)

    # write prediction to test_y file
    create_test_y_file(model4, test_loader)
