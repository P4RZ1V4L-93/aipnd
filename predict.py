from train import build_network
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models, io
from datetime import datetime
import json


import argparse

def load_checkpoint(filepath, device='cpu'):
    checkpoint = torch.load(filepath)
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    model, _, _ = build_network(device=device, structure=structure, hidden_layer1=hidden_layer1)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

# Process a PIL image for use in a PyTorch model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

    return transform(image)
    
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5, device='cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    image = Image.open(image_path)
    image = process_image(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        pred = torch.exp(model(image))
        pred = pred.topk(topk)
        
        classes = [list(model.class_to_idx.keys())[i] for i in pred.indices.cpu().numpy().squeeze()]
        return pred.values.cpu().numpy().squeeze(), classes
    
    
def view_classify(img, ps, classes, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    img = Image.open(img)
    cat = [cat_to_name[str(i)] for i in classes]
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), nrows=2)
    imshow(process_image(img), ax=ax1)
    ax2.barh(cat, ps)
    plt.tight_layout()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use a trained neural network')
    parser.add_argument('image_path', help='path to image')
    parser.add_argument('checkpoint', help='path to checkpoint')
    parser.add_argument('--top_k', default=5, type=int, help='top k most likely classes')
    parser.add_argument('--category_names', default='cat_to_name.json', help='category names')
    parser.add_argument('--gpu', default='cpu', help='enable gpu')

    args = parser.parse_args()

    image = args.image_path
    topk = args.top_k
    classes = args.category_names

    if args.gpu == 'gpu':
        if torch.cuda.is_available():
            print("GPU is available. Using GPU.")
            device = torch.device('cuda')
        else:
            print("GPU is not available. Using CPU instead.")
            device = torch.device('cpu')
    else:
        print("Using CPU.")
        device = torch.device('cpu')

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint, device=device)
    
    ps, classes = predict(image, model, topk, device=device)
    view_classify(image, ps, classes)