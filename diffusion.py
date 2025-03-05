import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import numpy as np
import wandb
import argparse
import os

# ---------------------
# 1) W&B Setup with Hyperparameter Configuration
# ---------------------
wandb.init(
    project="diffusion_mnist",
    entity="lucasarmand2-lucas-armand",
    config={
        "image_size": 28,
        "in_channels": 1,
        "out_channels": 1,
        "num_epochs": 100,
        "batch_size": 1024,
        "learning_rate": 1e-3,
        "T": 400,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "time_emb_dim": 128,
        "base_channels": 64,
        "clip": 1.0
    }
)
config = wandb.config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------
# 2) Data
# ---------------------
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

# ---------------------
# 3) Diffusion Utils
# ---------------------
def linear_beta_schedule(timesteps, start, end):
    return torch.linspace(start, end, timesteps)

T = config.T
beta_start = config.beta_start
beta_end = config.beta_end

betas = linear_beta_schedule(T, beta_start, beta_end).to(device)
alphas = 1. - betas
alpha_hats = torch.cumprod(alphas, dim=0)

def forward_diffusion_sample(x_0, t):
    sqrt_alpha_hat = torch.gather(torch.sqrt(alpha_hats), dim=0, index=t).reshape(-1, 1, 1, 1)
    sqrt_one_minus_alpha_hat = torch.gather(torch.sqrt(1 - alpha_hats), dim=0, index=t).reshape(-1, 1, 1, 1)
    eps = torch.randn_like(x_0)
    x_t = sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * eps
    return x_t, eps

# ---------------------
# 4) Model Definition: UNet Diffusion Model with Hyperparametric Controls
# ---------------------
class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(1, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    def forward(self, t):
        return self.embedding(t.unsqueeze(-1).float())

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x, t_emb):
        h = self.relu(self.conv1(x))
        # Incorporate time embedding via a FiLM-style addition
        time_emb = self.relu(self.time_emb_proj(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb
        h = self.relu(self.conv2(h))
        return h

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.block = UNetBlock(in_channels, out_channels, time_emb_dim)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x, t_emb):
        h = self.block(x, t_emb)
        p = self.pool(h)
        return h, p

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = UNetBlock(in_channels, out_channels, time_emb_dim)
        
    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x, t_emb)
        return x

class UNetDiffusionModel(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, base_channels):
        super().__init__()
        self.time_embedding = TimestepEmbedding(time_emb_dim)
        self.initial_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = DownBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.bottleneck = UNetBlock(base_channels * 4, base_channels * 8, time_emb_dim)
        self.up1 = UpBlock(base_channels * 8, base_channels * 4, time_emb_dim)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, time_emb_dim)
        self.final_conv = nn.Conv2d(base_channels * 2, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        x = self.initial_conv(x)
        skip1, x = self.down1(x, t_emb)
        skip2, x = self.down2(x, t_emb)
        x = self.bottleneck(x, t_emb)
        x = self.up1(x, skip2, t_emb)
        x = self.up2(x, skip1, t_emb)
        x = self.final_conv(x)
        return x

# ---------------------
# 5) Training Function
# ---------------------
def train(model, dataloader, num_epochs, save_path="model_weights.pth", clip=1.0):
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    mse = nn.MSELoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.to(device)
            t = torch.randint(0, T, (x.shape[0],), device=device).long()
            x_t, noise = forward_diffusion_sample(x, t)
            noise_pred = model(x_t, t)
            loss = mse(noise, noise_pred)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / (batch_idx + 1)
        duration = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Duration: {duration:.4f}s")
        wandb.log({"epoch": epoch+1, "loss": avg_loss, "duration": duration})
    
    # Save the model weights along with the config used in this run
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": dict(config)
    }, save_path)
    wandb.save(save_path)
    print(f"Model weights saved to {save_path}")

# ---------------------
# 6) Sampling Function
# ---------------------
@torch.no_grad()
def sample(model, n=16, save_path="generated_samples.png"):
    model.eval()
    x = torch.randn(n, config.in_channels, config.image_size, config.image_size, device=device)
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

    plt.figure(figsize=(4, 4))
    for i in range(n):
        plt.subplot(int(np.sqrt(n)), int(np.sqrt(n)), i+1)
        plt.imshow(x[i].cpu().squeeze(), cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved generated samples to {save_path}")

# ---------------------
# 7) Main Function
# ---------------------
def main():
    parser = argparse.ArgumentParser(description="Train or run inference with a diffusion model.")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], required=True, help="Mode: train or inference")

    # Add these new arguments (or any relevant ones)
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--T", type=int, default=400, help="Number of diffusion steps")
    parser.add_argument("--beta_start", type=float, default=1e-4, help="Beta start for linear schedule")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Beta end for linear schedule")
    parser.add_argument("--time_emb_dim", type=int, default=128, help="Dimensionality for time embeddings")
    parser.add_argument("--base_channels", type=int, default=64, help="Base channel count for UNet")
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--weights", type=str, default="model_weights.pth", help="Path to model weights for inference")

    args = parser.parse_args()

    # Now initialize wandb.config from these args:
    wandb.init(
        project="diffusion_mnist",
        entity="lucasarmand2-lucas-armand",
        config={
            "image_size": 28,
            "in_channels": 1,
            "out_channels": 1,
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "T": args.T,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "time_emb_dim": args.time_emb_dim,
            "base_channels": args.base_channels,
            "clip": args.clip
        }
    )
    config = wandb.config
    
    if args.mode == "inference":
        checkpoint = torch.load(args.weights, map_location=device)
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            loaded_config = checkpoint["config"]
        else:
            loaded_config = dict(config)
        
        model = UNetDiffusionModel(
            in_channels=loaded_config["in_channels"],
            out_channels=loaded_config["out_channels"],
            time_emb_dim=loaded_config["time_emb_dim"],
            base_channels=loaded_config["base_channels"]
        ).to(device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {args.weights}")
        sample(model, n=16)
    else:
        model = UNetDiffusionModel(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            time_emb_dim=config.time_emb_dim,
            base_channels=config.base_channels
        ).to(device)
        train(model, train_loader, args.epochs, save_path=args.weights, clip=args.clip)

if __name__ == "__main__":
    main()
