import os
import numpy as np
from networks import Generator,Discriminator
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch
from parameters import opt
from torchvision.utils import save_image

check=True
cuda = True if torch.cuda.is_available() else False

'''
traindata_path='/home/user/BCI_datasets/GAN_5000/IV_1_32*32/GAN_train_400/11'
checkpoint_path='./checkpoint/checkpoint/'
save_loss='./loss1.txt'
image_path='./images1/'
'''



traindata_path='/media/user/data/datasets/cifar10/cifar-10-python'
checkpoint_path='./CIFAR_checkpoint/'
image_path='./CIFAR_images/'
save_loss='./loss1.txt'

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()


# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# Configure data loader


#train_dataset=datasets.ImageFolder(traindata_path,transform)
train_dataset=datasets.CIFAR10(traindata_path,transform=transform)
dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=opt.batch_size,shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.glr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.dlr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
start_epoch = 0
if check:
    checkpoint=torch.load('./CIFAR_checkpoint/150')
    generator.load_state_dict(checkpoint['G'])
    discriminator.load_state_dict(checkpoint['D'])
    optimizer_G.load_state_dict(checkpoint['optimG'])
    optimizer_D.load_state_dict(checkpoint['optimD'])
    start_epoch = checkpoint['epoch']

for epoch in range(start_epoch+1,opt.n_epochs):
    #print(start_epoch)
    for i, (imgs, _) in enumerate(dataloader):
        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        gen_imgs = generator(z)
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ----------------------
        #  Train Discriminator
        # ----------------------

        optimizer_D.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss)/2
        d_loss.backward()
        optimizer_D.step()


    print(
        "[Epoch %d/%d]  [D loss: %f] [G loss: %f]"
        % (epoch, opt.n_epochs,  d_loss.item(), g_loss.item())
    )

    if epoch%30==0:
        # opt.lr=opt.lr/2
        checkpoint={
            'epoch':epoch,
            'G':generator.state_dict(),
            'D':discriminator.state_dict(),
            'optimG':optimizer_G.state_dict(),
            'optimD':optimizer_D.state_dict(),

        }
        torch.save(checkpoint, checkpoint_path + '%d' % (epoch))
    if epoch%3==0:
        with torch.no_grad():
            batch = 100
            z = torch.randn((batch, 100)).cuda()
            imgs = generator(z)
            save_image(imgs, image_path+'sonw%d.png'%(epoch), nrow=10,normalize=True)

