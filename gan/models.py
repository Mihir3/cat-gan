import torch


class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0)
        )
   

    def forward(self, x):

        x = self.model(x)

        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim

        self.model = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(noise_dim, 1024, kernel_size=4, stride=1, padding=0),
            torch.nn.BatchNorm2d(1024), 
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1),
            torch.nn.Tanh()
        )
    

    def forward(self, x):

        x = self.model(x)

        return x
