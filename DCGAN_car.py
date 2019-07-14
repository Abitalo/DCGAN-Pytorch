import torch.nn as nn
import config


class NetG(nn.Module):
    def __init__(self):
        super(NetG, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.ConvTranspose2d(config.latent_dim, 512, kernel_size=4, stride = 1, bias=False),
            nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True)
            nn.LeakyReLU(0.2, inplace=True)

        )

        self.layer_2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 5, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.Dropout(p=0.5),
#             nn.ReLU(inplace=True)
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer_3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 5, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.Dropout(p=0.5),
#             nn.ReLU(inplace=True)
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer_4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.Dropout(p=0.5),
#             nn.ReLU(inplace=True)
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer_5 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 5, 2, bias=False),
            nn.Tanh()
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)

        return out


class NetD(nn.Module):
    def __init__(self):
        super(NetD, self).__init__()
        self.layer_1 = nn.Sequential(   # 96*96*3
            nn.Conv2d(3, 128, kernel_size=5, stride=3, padding=1,  bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer_2 = nn.Sequential(   # 32*32*128
            nn.Conv2d(128, 256, 5, 2,  bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer_3 = nn.Sequential(   # 14*14*256
            nn.Conv2d(256, 256, 3, 2,  bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer_4 = nn.Sequential(   # 6*6*512
            nn.Conv2d(256, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer_5 = nn.Sequential(   # 3*3*1024
            nn.Conv2d(512, 1, 4, 1,  bias=False),
#             nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def forward(self, x):

        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        return out
