import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter
import os
import time


class Encoder(nn.Module):
    def __init__(self, out_shape, device):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=400, out_features=200),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=out_shape ** 2 + out_shape)
        )
        self.net.to(device)

    def forward(self, x):
        out = self.net(x)
        return out


class Decoder(nn.Module):
    def __init__(self, in_shape, device):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_shape, out_features=200),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=400),
            nn.ReLU()
        )
        self.ct = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=5),
            nn.Sigmoid()
        )
        self.fc.to(device)
        self.ct.to(device)

    def forward(self, z):
        z = self.fc(z)
        out = self.ct(z.view(z.shape[0], 1, 20, 20))
        return out

    def generate(self, z):
        score = self.forward(z)
        out = (score > 0.5).float()
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
        para = self.enc(x).view(x.shape[0], self.latent_shape + 1, self.latent_shape)
        ep = self.gaussian.sample(sample_shape=x.shape[0: 1]).view(-1, 1, self.latent_shape).to(self.device)
        latent = para[:, 0, :] + ep.matmul(para[:, 1:, :]).view(-1, self.latent_shape)
        out = self.dec(latent)
        return para, out

    def generate(self, x):
        _, score = self.forward(x)
        out = (score > 0.5).float()
        return out


def kl_gaussian(mean1, cov1, mean2, cov2):
    icov2 = torch.linalg.inv(cov2)  # torch.linalg.inv() not implemented in torch1.7.1+cu92
    ret = torch.log(cov2.det() / cov1.det()) - mean1.shape[-1] + torch.trace(icov2.matmul(cov1)) \
          + (mean2 - mean1).T.matmul(icov2).matmul(mean2 - mean1)
    return ret / 2


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
    writer = SummaryWriter(log_dir=log_dir)
    reconstruct = nn.BCELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
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
            optimizer.zero_grad()
            para, out = model.forward(data)
            loss1 = kl_gaussian(mean1=para[:, 0, :],
                                cov1=para[:, 1:, :],
                                mean2=torch.ones(latent_shape, device=device),
                                cov2=torch.eye(latent_shape, device=device))
            loss2 = reconstruct(out, data)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            epoch_loss[0] += loss1.to('cpu')
            epoch_loss[1] += loss2.to('cpu')
            writer.add_scalar('regularization', loss1, batch_idx)
            writer.add_scalar('reconstruction', loss2, batch_idx)
            writer.add_scalar('total', loss, batch_idx)
        duration = time.time() - start
        print('| epoch {} | KL {:.2f} | neg likelihood {:.2f} | total loss {:.2f} | training time {:.2f} s |'
              .format(epoch, epoch_loss[0], epoch_loss[1], epoch_loss.sum(), duration))

        if epoch % 5 == 0:
            print('Testing ...')
            start = time.time()
            test_dir = f'./test_images_{epoch}'
            if not os.path.isdir(test_dir):
                os.mkdir(test_dir)
            model.eval()
            for batch_idx, (data, _) in enumerate(test_loader):
                data = data.to(device)
                images = model.generate(data)
                img_name = f'image_{batch_idx}' + '.png'
                save_image(images.view(data.size(0), 1, 28, 28),
                           os.path.join(test_dir, img_name),
                           nrow=10)
            duration = time.time() - start
            print('images saved to {}, test time {:.2f} s'.format(test_dir, duration))


if __name__ == '__main__':
    main()
