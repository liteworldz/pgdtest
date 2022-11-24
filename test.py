import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import torchvision.models as models
import dataset
import time
import torchattacks


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
from models import *


filename = 'resnet18_NT.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

EPS= 8/255
ALPHA= 2/255
STEPS= 10

trainloader, valloader, testloader = dataset.get_loader()

net = models.resnet18()
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

load_path = "./checkpoint/"
checkpoint = torch.load(load_path + filename,
                        map_location=lambda storage, loc: storage.cuda(0))['net']
trainAccuracy = torch.load(load_path + filename,
                            map_location=lambda storage, loc: storage.cuda(0))['acc']
trainEpochs = torch.load(load_path + filename,
                            map_location=lambda storage, loc: storage.cuda(0))['epoch']

print('==> Loaded Model data..')
print("Train Acc", trainAccuracy)
print("Train Epochs", trainEpochs)
# Data
print('==> Preparing data..')
criterion = nn.CrossEntropyLoss()
print('\n[ Test Start ]')
start = time.time()

AUTOadversary = torchattacks.AutoAttack(
    net, norm='Linf', eps=EPS, version='standard', n_classes=10, seed=None, verbose=False)
PGDadversary = torchattacks.PGD(net, eps=EPS, alpha=ALPHA, steps=STEPS)

def test():
    print('\n[ Test Start ]')
    net.eval()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        benign_loss += loss.item()

        _, predicted = outputs.max(1)
        benign_correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0:
            print('\nCurrent batch:', str(batch_idx))
            print('Current benign test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current benign test loss:', loss.item())

        adv = PGDadversary(inputs, targets)
        adv_outputs = net(adv)
        loss = criterion(adv_outputs, targets)
        adv_loss += loss.item()

        _, predicted = adv_outputs.max(1)
        adv_correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0:
            print('Current adversarial test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current adversarial test loss:', loss.item())

    print('\nTotal benign test accuarcy:', 100. * benign_correct / total)
    print('Total adversarial test Accuarcy:', 100. * adv_correct / total)
    print('Total benign test loss:', benign_loss)
    print('Total adversarial test loss:', adv_loss)

test()
