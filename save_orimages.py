import numpy as np
from networks import Generator,Discriminator
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch
from parameters import opt
from torchvision.utils import save_image

traindata_path='/media/user/data/datasets/cifar10/cifar-10-python'
image_path='./orignal_image/'

transform=transforms.Compose([transforms.ToTensor(),
                              ])  #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
train_dataset=datasets.CIFAR10(traindata_path,transform=transform)
dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=opt.batch_size,shuffle=True)

for i, (imgs, _) in enumerate(dataloader):
    save_image(imgs, image_path + 'sonw%d.png' % (i+60), nrow=10)
    print(i)