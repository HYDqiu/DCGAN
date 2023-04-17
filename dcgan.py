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

traindata_path='/home/user/BCI_datasets/GAN_5000/IV_1_32*32/1'
checkpoint_path='./checkpoint/checkpoint4/'
save_loss='./loss3.txt'
image_path='./images4/'

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


train_dataset=datasets.ImageFolder(traindata_path,transform)

dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=opt.batch_size,shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
start_epoch = 1
if check:
    checkpoint=torch.load('./checkpoint/checkpoint4/100')
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
        if i % 2 == 0:
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

    if epoch%100==0:
        # opt.lr=opt.lr/2
        checkpoint={
            'epoch':epoch,
            'G':generator.state_dict(),
            'D':discriminator.state_dict(),
            'optimG':optimizer_G.state_dict(),
            'optimD':optimizer_D.state_dict(),
            'lr':opt.lr
        }
        torch.save(checkpoint, checkpoint_path + '%d' % (epoch))
    if epoch%200==0:
        with torch.no_grad():
            batch = 100
            z = torch.randn((batch, 100)).cuda()
            imgs = generator(z)
            save_image(imgs, image_path+'sonw%d.png'%(epoch), nrow=10)

