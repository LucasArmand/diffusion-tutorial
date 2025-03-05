from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import subprocess
import os

app = FastAPI()

# 1) Define a Pydantic model to specify what config fields you expect from a request.
class DiffusionConfig(BaseModel):
    mode: str
    epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 1e-3
    T: int = 400
    beta_start: float = 1e-4
    beta_end: float = 0.02
    time_emb_dim: int = 128
    base_channels: int = 64
    weights: str = "model_weights.pth"
    clip: float = 1.0


@app.post("/run_diffusion")
def run_diffusion(cfg: DiffusionConfig):
    """
    This endpoint spawns a subprocess to run the diffusion_mnist.py script
    with the config you sent in the POST request body.
    """
    # 2) Construct the command-line args for diffusion_mnist.py
    command = [
        "python", "diffusion.py",
        "--mode", cfg.mode,
        "--epochs", str(cfg.epochs),
        "--batch_size", str(cfg.batch_size),
        "--learning_rate", str(cfg.learning_rate),
        "--T", str(cfg.T),
        "--beta_start", str(cfg.beta_start),
        "--beta_end", str(cfg.beta_end),
        "--time_emb_dim", str(cfg.time_emb_dim),
        "--base_channels", str(cfg.base_channels),
        "--weights", cfg.weights,
        "--clip", str(cfg.clip)
    ]

    # 3) Run the script in a subprocess (blocking until it finishes).
    # For big training jobs, you might want an asynchronous approach or queue.
    result = subprocess.run(command, capture_output=True, text=True)

    # 4) Return the stdout/stderr so you can troubleshoot
    return {
        "stdout": result.stdout,
        "stderr": result.stderr
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
