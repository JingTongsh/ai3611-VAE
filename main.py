import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter
import os
import time


class Encoder(nn.Module):
    def __init__(self, out_shape, device):
        super(Encoder, self).__init__()
        self.out_shape = out_shape
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5),
            # nn.BatchNorm2d(num_features=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5),
            # nn.BatchNorm2d(num_features=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(in_features=400, out_features=200),
            nn.LeakyReLU(),
            nn.Linear(in_features=200, out_features=out_shape * 2)
        )
        self.net.to(device)

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.net(x).view(batch_size, 2, self.out_shape)
        mu = out[:, 0, :]
        sig = out[:, 1, :]
        return mu, sig


class Decoder(nn.Module):
    def __init__(self, in_shape, device):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_shape, out_features=200),
            nn.LeakyReLU(),
            nn.Linear(in_features=200, out_features=400),
            nn.LeakyReLU()
        )
        self.ct = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=5),
            # nn.BatchNorm2d(num_features=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=5),
            nn.Sigmoid()
        )
        self.fc.to(device)
        self.ct.to(device)

    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.shape[0], 1, 20, 20)
        out = self.ct(z)
        return out

    def generate(self, z):
        out = self.forward(z)
        # out = (out > 0.5).float()
        return out


class VAE(nn.Module):
    def __init__(self, latent_shape, device):
        super().__init__()
        self.latent_shape = latent_shape
        self.device = device
        self.enc = Encoder(out_shape=latent_shape, device=device)
        self.dec = Decoder(in_shape=latent_shape, device=device)
        self.gaussian = MultivariateNormal(loc=torch.zeros(self.latent_shape),
                                           covariance_matrix=torch.eye(self.latent_shape))

    def forward(self, x):
        mu, sig = self.enc(x)
        ep = self.gaussian.sample(sample_shape=x.shape[0: 1]).to(self.device)
        latent = mu + sig * ep
        out = self.dec(latent)
        return mu, sig, out

    def generate_from_img(self, x):
        _, _, out = self.forward(x)
        # out = (out > 0.5).float()
        return out

    def generate_from_latent(self, z):
        out = self.dec(z)
        return out


def main():
    # Arguments
    batch_size = 16
    device = 'cuda:0'
    epochs = 10
    latent_shape = 1
    learning_rate = 1e-3
    log_dir = './'
    save_dir = './'

    # Download MNIST Dataset
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=False)

    # MNist Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize
    model = VAE(latent_shape=latent_shape, device=device)
    model.to(device)
    # writer = SummaryWriter(log_dir=log_dir)
    optim = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
    target = torch.zeros(batch_size, latent_shape + 1, latent_shape, requires_grad=False, device=device)
    target[1:, :] = torch.eye(latent_shape)

    # Train
    print('Training ...')
    for epoch in range(1, epochs + 1):
        start = time.time()
        epoch_loss = torch.zeros(2)

        for batch_idx, (data, _) in enumerate(train_loader):
            model.train()
            data = data.to(device)
            optim.zero_grad()
            mu, sig, out = model.forward(data)
            sig2 = sig ** 2

            kl_divergence = -(1 + torch.log(sig2) - mu ** 2 - sig2).sum() / 2
            rec_loss = F.mse_loss(out, data)
            loss = kl_divergence + rec_loss
            loss.backward()
            optim.step()

            epoch_loss[0] += kl_divergence.to('cpu')
            epoch_loss[1] += rec_loss.to('cpu')

            # writer.add_scalar('regularization', loss1, batch_idx)
            # writer.add_scalar('reconstruction', loss2, batch_idx)
            # writer.add_scalar('total', loss, batch_idx)

        duration = time.time() - start
        print('| epoch {} | KL {:.2f} | neg likelihood {:.2f} | total loss {:.2f} | training time {:.2f} s |'
              .format(epoch, epoch_loss[0], epoch_loss[1], epoch_loss.sum(), duration))

        if epoch % 1 == 0:
            print('Testing ...')
            start = time.time()
            test_dir = f'./test_images_{epoch}'
            if not os.path.isdir(test_dir):
                os.mkdir(test_dir)
            model.eval()
            for batch_idx, (data, _) in enumerate(test_loader):
                data = data.to(device)
                images = model.generate_from_img(data)
                img_name = f'image_{batch_idx}' + '.png'
                save_image(images.view(data.size(0), 1, 28, 28),
                           os.path.join(test_dir, img_name),
                           nrow=8)
            duration = time.time() - start
            print('images saved to {}/; test time {:.2f} s'.format(test_dir, duration))


if __name__ == '__main__':
    main()
