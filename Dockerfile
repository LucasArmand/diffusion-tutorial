# Use a CUDA-enabled PyTorch image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set the working directory
WORKDIR /app

# Copy requirements file (if you prefer managing deps this way)
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the source code into the container
COPY . .

# The container will run the training script.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

