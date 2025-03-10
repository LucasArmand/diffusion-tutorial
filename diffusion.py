import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import numpy as np
import wandb

# ---------------------
# 1) W&B Setup
# ---------------------
wandb.init(project="diffusion_mnist", entity="lucasarmand2-lucas-armand")  # set your wandb entity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------
# 2) Config
# ---------------------
image_size = 28
channels = 1
num_epochs = 100
batch_size = 1024
learning_rate = 1e-3
T = 400
beta_start = 1e-4
beta_end = 0.02

# ---------------------
# 3) Data
# ---------------------
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ---------------------
# 4) Diffusion Utils
# ---------------------
def linear_beta_schedule(timesteps, start=beta_start, end=beta_end):
    return torch.linspace(start, end, timesteps)

betas = linear_beta_schedule(T, beta_start, beta_end).to(device)
alphas = 1. - betas
alpha_hats = torch.cumprod(alphas, dim=0)

def forward_diffusion_sample(x_0, t):
    sqrt_alpha_hat = torch.gather(torch.sqrt(alpha_hats), dim=0, index=t).reshape(-1,1,1,1)
    sqrt_one_minus_alpha_hat = torch.gather(torch.sqrt(1-alpha_hats), dim=0, index=t).reshape(-1,1,1,1)
    eps = torch.randn_like(x_0)
    x_t = sqrt_alpha_hat*x_0 + sqrt_one_minus_alpha_hat*eps
    return x_t, eps

# ---------------------
# 5) Model Definition
# ---------------------
class TimestepEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(1, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    def forward(self, t):
        return self.embedding(t.unsqueeze(-1).float())

class SimpleDiffusionModel(nn.Module):
    def __init__(self, time_emb_dim=64):
        super().__init__()
        self.time_embedding = TimestepEmbedding(dim=time_emb_dim)
        self.conv1 = nn.Conv2d(channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.deconv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.deconv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.output = nn.Conv2d(32, channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.fc_time1 = nn.Linear(time_emb_dim, 64)
        self.fc_time2 = nn.Linear(time_emb_dim, 128)
        self.fc_time3 = nn.Linear(time_emb_dim, 64)
        
    def forward(self, x, t):
        t_embed = self.time_embedding(t)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x) + self.fc_time1(t_embed).unsqueeze(-1).unsqueeze(-1))
        x = self.relu(self.conv3(x) + self.fc_time2(t_embed).unsqueeze(-1).unsqueeze(-1))
        x = self.relu(self.deconv1(x) + self.fc_time3(t_embed).unsqueeze(-1).unsqueeze(-1))
        x = self.relu(self.deconv2(x))
        x = self.output(x)
        return x

model = SimpleDiffusionModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
mse = nn.MSELoss()

# ---------------------
# 6) Training Loop
# ---------------------
def train_one_epoch(model, dataloader):
    model.train()
    total_loss = 0
    for batch_idx, (x, _) in enumerate(dataloader):
        x = x.to(device)
        t = torch.randint(0, T, (x.shape[0],), device=device).long()
        x_t, noise = forward_diffusion_sample(x, t)
        noise_pred = model(x_t, t)
        loss = mse(noise, noise_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / (batch_idx + 1)
    return avg_loss

for epoch in range(num_epochs):
    start_time = time.time()
    avg_loss = train_one_epoch(model, train_loader)
    duration = time.time() - start_time
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Duration: {duration:.4f}s")
    wandb.log({"epoch": epoch+1, "loss": avg_loss, "duration": duration})

# Save the model weights for later local sampling
torch.save(model.state_dict(), "model_weights.pth")
wandb.save("model_weights.pth")

# ---------------------
# 7) Sampling & Saving Visualizations
# ---------------------
@torch.no_grad()
def sample(model, n=16):
    model.eval()
    x = torch.randn(n, channels, image_size, image_size, device=device)
    for i in reversed(range(T)):
        t = (torch.ones(n) * i).long().to(device)
        eps_theta = model(x, t)
        beta_t = betas[i]
        alpha_t = alphas[i]
        alpha_hat_t = alpha_hats[i]
        z = torch.randn_like(x) if i > 0 else 0
        x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * eps_theta)
        if i > 0:
            x = x + torch.sqrt(beta_t) * z
    x = torch.clamp(x, -1., 1.)
    x = (x + 1) / 2
    return x

samples = sample(model, n=16)

# Instead of plt.show(), save the samples as an image file.
plt.figure(figsize=(4, 4))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(samples[i].cpu().squeeze(), cmap="gray")
    plt.axis("off")
plt.tight_layout()
plt.savefig("samples.png")
print("Saved generated samples to samples.png")

