# MK-UNet Extension of Vision Mamba

#### Quick Run: 
```
conda env create -f environment.yml
conda activate mku
```

##### Dependencies:

```

# install Anaconda from https://www.anaconda.com/products/distribution
# then open terminal or Anaconda Prompt

conda create -n mku python=3.10 -y        # create new environment named 'mku' with Python 3.10
conda activate mku                        # activate the environment

# install PyTorch + torchvision + torchaudio (for CUDA 12.1)
pip install torch torchvision torchaudio

# or use this line instead if you want CPU-only PyTorch
# pip install torch torchvision torchaudio

pip install timm                          # install timm library (PyTorch image models)
pip install numpy matplotlib tqdm opencv-python  # optional but useful dependencies
pip install vision-mamba
pip install datasets


python -c "import torch, timm; print(torch.__version__, timm.__version__)"  # verify installations
```
