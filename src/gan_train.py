import os
import random
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

# ==== Configuration ====
DATASET_DIR = r"C:\Users\neash\Documents\Research Papers\VIPeR.v1.0\VIPeR"
CAM_FOLDER = 'cam_a'  # You can use 'cam_b' or both if needed
IMAGE_SIZE = 64       # Raise to 128 or 224 for higher detail (recommended only with a good GPU and large batch size)
BATCH_SIZE = 32
LATENT_DIM = 100
EPOCHS = 200
CRITIC_ITER = 5      # WGAN-GP: update D multiple times per G update
LAMBDA_GP = 10
OUTPUT_DIR = r"C:\Users\neash\Documents\Research Papers\VIPeR.v1.0\VIPeR\generated_viper_images"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== Dataset Loader ====
class VIPeRDataset(Dataset):
    def __init__(self, root_dir, cam_folder='cam_a', transform=None):
        self.img_dir = os.path.join(root_dir, cam_folder)
        self.img_paths = glob.glob(os.path.join(self.img_dir, '*.bmp'))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = VIPeRDataset(DATASET_DIR, cam_folder=CAM_FOLDER, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# ==== Spectral Norm wrapper for stable D ===
def sn(module):
    return nn.utils.spectral_norm(module)

# ==== Generator ====
class Generator(nn.Module):
    def __init__(self, latent_dim, channels=3, features_g=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, features_g*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features_g*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g*8, features_g*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g*4, features_g*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g*2, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)

# ==== Discriminator (Critic) ====
class Discriminator(nn.Module):
    def __init__(self, channels=3, features_d=64):
        super().__init__()
        self.main = nn.Sequential(
            sn(nn.Conv2d(channels, features_d, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Conv2d(features_d, features_d*2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(features_d*2),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Conv2d(features_d*2, features_d*4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(features_d*4),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Conv2d(features_d*4, features_d*8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(features_d*8),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Conv2d(features_d*8, 1, 4, 1, 0, bias=False))
        )

    def forward(self, x):
        return self.main(x).view(-1)

def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def gradient_penalty(critic, real, fake, device=DEVICE):
    BATCH_SIZE = real.size(0)
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1), device=device, requires_grad=True)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)
    mixed_scores = critic(interpolated)
    grad_outputs = torch.ones_like(mixed_scores, device=device)
    gradients = torch.autograd.grad(
        inputs=interpolated, outputs=mixed_scores, grad_outputs=grad_outputs,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    random.seed(42)
    torch.manual_seed(42)

    netG = Generator(LATENT_DIM).to(DEVICE)
    netD = Discriminator().to(DEVICE)
    netG.apply(weights_init)
    netD.apply(weights_init)

    optimizerD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.0, 0.9))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.0, 0.9))

    for epoch in range(EPOCHS):
        for i, real_imgs in enumerate(dataloader):
            real_imgs = real_imgs.to(DEVICE)
            B = real_imgs.size(0)

            # === Train Discriminator (Critic) ===
            for _ in range(CRITIC_ITER):
                z = torch.randn(B, LATENT_DIM, 1, 1, device=DEVICE)
                fake_imgs = netG(z)
                D_real = netD(real_imgs)
                D_fake = netD(fake_imgs.detach())
                gp = gradient_penalty(netD, real_imgs, fake_imgs, DEVICE)

                loss_D = -(torch.mean(D_real) - torch.mean(D_fake)) + LAMBDA_GP * gp

                optimizerD.zero_grad()
                loss_D.backward()
                optimizerD.step()

            # === Train Generator ===
            z = torch.randn(B, LATENT_DIM, 1, 1, device=DEVICE)
            fake_imgs = netG(z)
            loss_G = -torch.mean(netD(fake_imgs))

            optimizerG.zero_grad()
            loss_G.backward()
            optimizerG.step()

        # --- Save a batch of individual images after each epoch ---
        for idx in range(fake_imgs.size(0)):
            image_out = fake_imgs[idx]
            save_path = os.path.join(OUTPUT_DIR, f"epoch{epoch+1}_img{idx+1}.png")
            save_image(image_out, save_path, normalize=True)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | D Loss: {loss_D.item():.3f} | G Loss: {loss_G.item():.3f}")

    print(f"Training done! Generated images are in '{OUTPUT_DIR}'.")

