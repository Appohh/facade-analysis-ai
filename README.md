# facade-analysis-ai
Project that analyzes facades images with segmentation


--------------------------------------------------------

# Installing Dependencies
To be able to run the notebooks and scripts in this repository there are a specific type of dependencies that need to be installed with specific versions

The models in this project are trained with GPU Acceleration using Cuda 11.2 with CUDNN v8.1.1.33
Python 3.10
Tensorflow-gpu 2.10.0
Torch 2.7.0+cu118



# Steps
In order to install dependencies without version conflicts a few dependencies need to be installed in a order within the environment

## Step 0
Create a virtual environment with the command: "python -m venv venv" in the terminal at the root directory of the project.
And activate it. Run: "venv\Scripts\activate" 

## Step 1
Install a specific version of PyTorch by index
Run: "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"

## Step 2
Install a Numpy version below 2.0
Run: pip install "numpy<2.0"

## Step 3
Install Tensorflow-gpu 2.10.0
Run: "pip install tensorflow-gpu==2.10.0"

## Step 4 Install remaining requirements
Run: "pip install -r requirements.txt"

--------------------------------------------------------
