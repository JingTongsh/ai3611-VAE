import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import os
import time


def get_model(shape: list, name: str) -> list:
    assert len(shape) > 0, shape
    assert name.lower() in ['conv', 'linear', 'conv_transpose'], name

    model = []
    if name.lower() == 'conv':
        for k in range(len(shape) - 1):
            model += [nn.Conv2d(in_channels=shape[k], out_channels=shape[k + 1], kernel_size=5),
                      nn.BatchNorm2d(num_features=shape[k + 1]),
                      nn.Tanh()]
    elif name.lower() == 'linear':
        for k in range(len(shape) - 1):
            model += [nn.Linear(in_features=shape[k], out_features=shape[k+1]),
                      nn.Tanh()]
    else:
        for k in range(len(shape) - 1):
            model += [nn.ConvTranspose2d(in_channels=shape[k], out_channels=shape[k + 1], kernel_size=5),
                      nn.BatchNorm2d(num_features=shape[k + 1]),
                      nn.Tanh()]

    return model


class MyVAE(nn.Module):
    def __init__(self, latent_dim: int = 1):
        super(MyVAE, self).__init__()
        # encoder
        channels = [1, 6, 6, 12, 12]
        feature_dims = [12*12*12, 240, latent_dim]
        model = get_model(channels, 'conv') + [nn.Flatten()] + get_model(feature_dims, 'linear')
        self.encoder = nn.Sequential(*model)

        # decoder
        channels.reverse()
        feature_dims.reverse()
        model = get_model(feature_dims, 'linear')
        self.decoder_fc = nn.Sequential(*model)
        model = get_model(channels, 'conv_transpose')
        self.decoder_ct = nn.Sequential(*model)

    def forward(self, x):
        z = self.encoder(x)
        temp = self.decoder_fc(z)
        temp = temp.view(z.shape[0], 12, 12, 12)
        out = self.decoder_ct(temp)
        return z, out

    def generate_from_image(self, x):
        _, out = self.forward(x)
        # out = (out > 0.5).float()
        return out

    def generate_from_latent(self, z):
        temp = self.decoder_fc(z)
        temp = temp.view(z.shape[0], 12, 12, 12)
        out = self.decoder_ct(temp)
        # out = (out > 0.5).float()
        return out


def simple_log(f, info):
    f.write(info + '\n')
    print(info)


def do_one_epoch(model, loader, device, writer, log_file, epoch, train: bool, optim=None, test_dir=None):
    mode = 'train' if train else 'test'
    if train:
        assert optim is not None
    else:
        assert test_dir is not None

    start = time.time()
    epoch_loss = torch.zeros(2)
    model.train()

    for batch_idx, (data, _) in enumerate(loader):
        data = data.to(device)
        if train:
            optim.zero_grad()
        latent, out = model.forward(data)
        mu = latent.mean(dim=0)
        sig = latent.T.cov()  # in that `cov()` takes columns as samples
        if sig.ndim < 2:
            sig = sig.view(1, 1)
        kl_div = (-torch.log(sig.det()) - latent.shape[-1] + sig.trace() + mu.matmul(mu)) / 2
        rec_loss = F.mse_loss(out, data)
        loss = kl_div + rec_loss
        if train:
            loss.backward()
            optim.step()
        else:
            img_name = f'image_{batch_idx}' + '.png'
            save_image(out.view(data.size(0), 1, 28, 28),
                       os.path.join(test_dir, img_name),
                       nrow=8)

        epoch_loss += torch.Tensor([kl_div, rec_loss], device='cpu')

    duration = time.time() - start

    # Tensorboard writer
    epoch_kl, epoch_rc = epoch_loss
    epoch_total = epoch_loss.sum()
    writer.add_scalar(f'{mode} KL divergence', epoch_kl, epoch)
    writer.add_scalar(f'{mode} reconstruction loss', epoch_rc, epoch)
    writer.add_scalar(f'{mode} train loss', epoch_total, epoch)

    # log
    log_info = '| epoch {} | {} | kl {:.2f} | reconstruction {:.2f} | total loss {:.2f} | time {:.2f} s |' \
        .format(epoch, mode, epoch_kl, epoch_rc, epoch_total, duration)
    simple_log(log_file, log_info)


def main():
    # Arguments
    batch_size = 16
    device = 'cuda:0'
    do_train = False
    do_generate = True
    epochs = 20
    latent_dim = 2
    learning_rate = 1e-2
    model_save_dir = './model'
    test_dir = f'./test_images_D_{latent_dim}'
    generate_dir = f'./generate_D_{latent_dim}'

    # Preparations
    if not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)

    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)

    if not os.path.isdir(generate_dir):
        os.mkdir(generate_dir)

    log_file = open('log.txt', 'w')

    # Download MNIST Dataset
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=False)

    # MNist Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize
    model = MyVAE(latent_dim=latent_dim)
    model.to(device)
    writer = SummaryWriter(log_dir='./')
    optim = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

    # Train and test
    if do_train:
        print('Model:\n', model)
        print('Training ...')
        for epoch in range(1, epochs + 1):
            # train
            do_one_epoch(model, train_loader, device, writer, log_file, epoch, train=True, optim=optim)
            # save
            torch.save(model, os.path.join(model_save_dir, f'model_D_{latent_dim}.pt'))
            # test
            do_one_epoch(model, test_loader, device, writer, log_file, epoch, train=False, test_dir=test_dir)
    else:
        model = torch.load(os.path.join(model_save_dir, f'model_D_{latent_dim}.pt'))

    # Generate images
    if do_generate:
        ts = []
        for k in range(latent_dim):
            ts.append(torch.linspace(-5., 5., steps=9))
        z = torch.cartesian_prod(*ts)
        if latent_dim == 1:
            z = z.unsqueeze(-1)
        z = z.to(device)
        images = model.generate_from_latent(z)
        save_image(images.view(-1, 1, 28, 28),
                   os.path.join(generate_dir, 'image.png'),
                   nrow=9)


if __name__ == '__main__':
    main()
