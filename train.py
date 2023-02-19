# Imports here
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models, io
from datetime import datetime

import argparse

def build_network(device='cpu', structure='resnet50', hidden_layer1=[1000, 500], lr=0.001):
    if structure == 'resnet50':
        model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        input_size = 2048
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential()
        classifier.append(nn.Linear(input_size, hidden_layer1[0]))
        classifier.append(nn.ReLU())
        for i in range(len(hidden_layer1)-1):
            classifier.append(nn.Linear(hidden_layer1[i], hidden_layer1[i+1]))
            classifier.append(nn.ReLU())
        classifier.append(nn.Linear(hidden_layer1[-1], 102))
        classifier.append(nn.LogSoftmax(dim=1))
            
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    elif structure == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential()
        classifier.append(nn.Linear(25088, hidden_layer1[0]))
        classifier.append(nn.ReLU())
        for i in range(len(hidden_layer1)-1):
            classifier.append(nn.Linear(hidden_layer1[i], hidden_layer1[i+1]))
            classifier.append(nn.ReLU())
        classifier.append(nn.Linear(hidden_layer1[-1], 102))
        classifier.append(nn.LogSoftmax(dim=1))
            
        model.classifier = classifier
        optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    else:
        raise SystemExit("Invalid model architecture. Please choose from 'resnet50' or 'vgg16'")

    model.to(device)
    criterion = nn.NLLLoss()
    return model, optimizer, criterion

def train_network(model, optimizer, criterion, epochs=10, print_every=5):
    steps = 0
    running_loss = 0
    print("Training started at: ", datetime.now())
    for epoch in range(epochs):
        tot = datetime.now()
        for inputs, labels in trainloader:
            start = datetime.now()
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        test_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                n = len(testloader)
                finish = datetime.now()
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"steps {steps}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Val loss: {test_loss/n:.3f}.. "
                    f"Val accuracy: {accuracy/n:.3f}.. "
                    f"time taken: {(finish-start).seconds}s.. "
                    f"epoch time taken: {(finish-tot).seconds}s.. ")
                running_loss = 0
                model.train()
    print("Training finished at: ", datetime.now())

def check_accuracy_on_test(model, testloader):    
    correct = 0
    total = 0
    model.to('cuda:0')
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    
def save_checkpoint(model, structure='resnet50', hidden_layer1=[1000, 500], save_dir=''):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'structure': structure,
                  'hidden_layer1': hidden_layer1,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, save_dir+'/checkpoint.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument('data_dir', help='path to data directory')
    parser.add_argument('--arch', default='resnet50', help='architecture')
    parser.add_argument('--save_dir', default='.', help='save directory')
    parser.add_argument('--learning_rate', default=0.001, help='learning rate')
    parser.add_argument('--hidden_units', default=[1000, 500], help='hidden units')
    parser.add_argument('--epochs', default=10, help='epochs')
    parser.add_argument('--gpu', default='cpu', help='enable gpu')

    n = parser.parse_args()
    data_dir = n.data_dir
    hidden_layer1 = list(map(int, n.hidden_units.split(',')))

    if n.gpu == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("GPU is not available. Using CPU instead.")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    model, optimizer, criterion = build_network(device=device,structure=n.arch, hidden_layer1=hidden_layer1, lr=float(n.learning_rate))

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Define transforms for the training data and testing data
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    # Pass transforms in here
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    train_network(model, optimizer, criterion, epochs=int(n.epochs), print_every=5)

    check_accuracy_on_test(model, testloader)

    save_checkpoint(model, structure=n.arch, hidden_layer1=hidden_layer1, save_dir=n.save_dir)
