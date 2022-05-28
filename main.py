import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import os
import time


class SimpleVAE(nn.Module):
    def __init__(self, latent_shape):
        super(SimpleVAE, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.Tanh(),
            nn.Conv2d(in_channels=6, out_channels=6, kernel_size=5),
            nn.Tanh(),
            nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(in_features=6*18*18, out_features=1200),
            nn.Tanh(),
            nn.Linear(in_features=1200, out_features=latent_shape)
        )
        self.dec_fc = nn.Sequential(
            nn.Linear(in_features=latent_shape, out_features=1200),
            nn.Tanh(),
            nn.Linear(in_features=1200, out_features=6*18*18),
            nn.Tanh(),
        )
        self.dec_cv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=6, out_channels=6, kernel_size=3),
            nn.Tanh(),
            nn.ConvTranspose2d(in_channels=6, out_channels=6, kernel_size=5),
            nn.Tanh(),
            nn.ConvTranspose2d(in_channels=6, out_channels=1, kernel_size=5),
            nn.Tanh(),
        )

    def forward(self, x):
        z = self.enc(x)
        temp = self.dec_fc(z)
        temp = temp.view(z.shape[0], 6, 18, 18)
        out = self.dec_cv(temp).view(z.shape[0], 1, 28, 28)
        return z, out

    def generate_from_img(self, x):
        _, out = self.forward(x)
        # out = (out > 0.5).float()
        return out

    def generate_from_latent(self, z):
        out = self.dec(z)
        return out


def simple_log(f, info):
    f.write(info + '\n')
    print(info)


def main():
    # Arguments
    batch_size = 32
    device = 'cuda:0'
    epochs = 10
    latent_shape = 2
    learning_rate = 1e-3
    model_save_dir = './model'

    if not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)

    log_file = open('log.txt', 'w')

    # Download MNIST Dataset
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=False)

    # MNist Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize
    model = SimpleVAE(latent_shape=latent_shape)
    model.to(device)
    writer = SummaryWriter(log_dir='./')
    optim = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

    # Train
    print('Training ...')
    for epoch in range(1, epochs + 1):
        start = time.time()
        epoch_loss = torch.zeros(2)

        for batch_idx, (data, _) in enumerate(train_loader):
            model.train()
            data = data.to(device)
            optim.zero_grad()
            latent, out = model.forward(data)
            mu = latent.mean(dim=0)
            sig = latent.T.cov()
            if sig.ndim < 2:
                sig = sig.view(1, 1)
            kl_div = (-torch.log(sig.det()) - latent.shape[-1] + sig.trace() + mu.matmul(mu)) / 2
            rec_loss = F.mse_loss(out, data)
            loss = kl_div + rec_loss
            loss.backward()
            optim.step()

            epoch_loss += torch.Tensor([kl_div, rec_loss], device='cpu')

        ep_kld, ep_rec = epoch_loss
        ep_loss = epoch_loss.sum()
        writer.add_scalar('KL', ep_kld, epoch)
        writer.add_scalar('reconstruction', ep_rec, epoch)
        writer.add_scalar('total', ep_loss, epoch)

        duration = time.time() - start
        log_info = '| epoch {} | kl divergence {:.2f} | rec loss {:.2f} | total loss {:.2f} | training time {:.2f} s |'\
                   .format(epoch, ep_kld, ep_rec, ep_loss, duration)
        simple_log(log_file, log_info)

        if epoch % 5 == 0:
            print('Testing ...')
            start = time.time()
            test_dir = f'./test_images_{latent_shape}_{epoch}'
            if not os.path.isdir(test_dir):
                os.mkdir(test_dir)
            model.eval()

            # generate and save images
            for batch_idx, (data, _) in enumerate(test_loader):
                data = data.to(device)
                images = model.generate_from_img(data)
                img_name = f'image_{batch_idx}' + '.png'
                save_image(images.view(data.size(0), 1, 28, 28),
                           os.path.join(test_dir, img_name),
                           nrow=8)
            duration = time.time() - start
            log_info = 'generated test images saved to {}/; test time {:.2f} s'.format(test_dir, duration)
            simple_log(log_file, log_info)

            # save model
            torch.save(model, os.path.join(model_save_dir, f'model_{latent_shape}_{epoch}.pt'))


if __name__ == '__main__':
    main()
