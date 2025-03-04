# Use a CUDA-enabled PyTorch image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy requirements file (if you prefer managing deps this way)
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the source code into the container
COPY . .

# The container will run the training script.
CMD ["python", "train.py"]

