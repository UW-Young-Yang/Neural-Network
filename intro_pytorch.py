import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_set=datasets.FashionMNIST('./data',train=True,
        download=True,transform=transform)
    test_set=datasets.FashionMNIST('./data', train=False,
        transform=transform)
    if training:
        return torch.utils.data.DataLoader(train_set, batch_size=64)
    else:
        return torch.utils.data.DataLoader(test_set, batch_size=64)
    



def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    return model



def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(T):
        running_loss = 0
        accuracy = 0
        for images, labels in train_loader:
            optimizer.zero_grad()   # zero the parameter gradients
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()    # Does the update
            
            running_loss += loss.item()
            accuracy += (outputs.argmax(axis=1) == labels).sum()
        len_dataset = len(train_loader.dataset) # length of train dataset: 60000
        len_loader = len(train_loader) # the number of batches: 938
        accuracy_rate = (accuracy / len_dataset) * 100
        loss_rate = running_loss/len_loader
        print(f'Train Epoch: {epoch} Accuracy: {accuracy}/{len_dataset}({accuracy_rate:.2f}%) Loss: {loss_rate:.3f}')
        
    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    with torch.no_grad():
        accuracy = 0
        running_loss = 0
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            accuracy += (outputs.argmax(axis=1) == labels).sum()
        len_dataset = len(test_loader.dataset)
        len_loader = len(test_loader)
        accuracy_rate = (accuracy / len_dataset) * 100
        loss_rate = running_loss/len_loader
    if show_loss:
        print(f'Average loss: {loss_rate:.4f}')
    print(f'Accuracy: {accuracy_rate:.2f}%')


    


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt'
                    ,'Sneaker','Bag','Ankle Boot']
    test_image = test_images[index]
    output = model(test_image)[0]
    prob = F.softmax(output, 0)
    rank = torch.argsort(prob, descending=True)
    for i in range(3): # the top three
        name = class_names[rank[i]]
        proba = prob[rank[i]] * 100
        print(f'{name}: {proba:.2f}%')




if __name__ == '__main__':
    '''
    Feel free to write your own test code here to examine the correctness of your functions. 
    Note that this part will not be graded.
    '''
    # criterion = nn.CrossEntropyLoss()
    # train_loader = get_data_loader()
    # print(type(train_loader))
    # print(train_loader.dataset)
    # test_loader = get_data_loader(False)
    # print(type(test_loader))
    # print(test_loader.dataset)

    # model = build_model()
    # print(model)
    # for i in range(6):
    #     print(list(model.parameters())[i].size())

    # # print(list(train_loader)[0][0].shape)    # torch.Size([64, 1, 28, 28])
    # train_model(model, train_loader, criterion, 5)
    # evaluate_model(model, test_loader, criterion, show_loss = True)

    # pred_set = list(test_loader)[0][0]
    # predict_label(model, pred_set, 1)