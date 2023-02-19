import argparse
import sys

parser = argparse.ArgumentParser(description='Train a neural network')
parser.add_argument('')
parser.add_argument('--arch', default='resnet50', help='architecture')


n = parser.parse_args()
print(n.arch)
print(n.noarg)