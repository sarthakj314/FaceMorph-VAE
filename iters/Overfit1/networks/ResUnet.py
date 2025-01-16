import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, padding='same')
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, padding='same')
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.downsize = None
        self.bn3 = None
        if(in_dim != out_dim):
            self.downsize = nn.Conv2d(in_dim, out_dim, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_dim)

    
    def forward(self, x):
        out = self.conv1(x)
        out = nn.ReLU()(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = nn.ReLU()(out)
        out = self.bn2(out)
        if self.downsize != None and self.bn3 != None:
            x = self.downsize(x)
            x = self.bn3(x)
        out += x
        return out

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(1, 16), 
            Block(16, 32), 
            Block(32, 64), 
            Block(64, 128)])
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.avgpool = nn.AvgPool2d(4)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()

        self.kl = None

    def forward(self, x):
        for idx in range(len(self.blocks)):
            x = self.blocks[idx](x)
            if idx != len(self.blocks)-1:
                x = self.maxpool(x)
        x = self.avgpool(x)
        x = torch.squeeze(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        std = torch.exp(.5 * logvar)
        self.kl = -.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        z = std*self.N.sample(mu.shape) + mu
        z = z.unsqueeze(-1).unsqueeze(-1)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.tconv = nn.ModuleList([
            nn.ConvTranspose2d(latent_dim, latent_dim, 4),
            nn.ConvTranspose2d(128, 64, 4),
            nn.ConvTranspose2d(32, 32, 2, 2),
            nn.ConvTranspose2d(16, 4, 2, 2)])
        self.bn = nn.ModuleList([
            nn.BatchNorm2d(latent_dim),
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(32),
            nn.BatchNorm2d(4)])
        self.blocks = nn.ModuleList([
            Block(latent_dim, 128),
            Block(64, 32),
            Block(32, 16),
            Block(4, 1)])

    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.tconv[i](x)
            x = self.bn[i](x)
            x = self.blocks[i](x)
        x = torch.sigmoid(x)
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
