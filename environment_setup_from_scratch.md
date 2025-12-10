# Terminal Commands for Environment Setup from Scratch

These terminal commands to setup the environment from scratch, were communicated to us by an external user of our repo. We highly appreciate your feedback!
Cuda toolkit and pytorch versions might have to be adjusted according to your ressources. Compatible versions can be found on https://pytorch.org/get-started/previous-versions/.<img width="395" height="21" alt="image" src="https://github.com/user-attachments/assets/d213465a-a774-4d23-b073-464a580340ca" />


```bash
conda create -n inr-env python=3.10 pip -c conda-forge
```

```bash
conda activate inr-env
```

```bash
conda config --env --set channel_priority strict
```

```bash
conda install -c conda-forge \
   numpy=1.24 \
   scipy \
   sigpy \
   nibabel \
   scikit-image \
   h5py \   matplotlib \
   pyyaml \
   tqdm \   cmake \   ninja \   gcc_linux-64 gxx_linux-64
```

```bash
conda install -c conda-forge cudatoolkit=11.8
```

```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

```bash
nvcc -V
```

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
   --extra-index-url https://download.pytorch.org/whl/cu118
```

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

```bash
pip install \
   hydra-core==1.3.2 \
   omegaconf \   wandb \   medutils-mri \   pydicom
```

```bash
conda install -c conda-forge "gcc_linux-64=11.4.0" "gxx_linux-64=11.4.0" "gfortran_linux-64=11.4.0"
```

```bash
which x86_64-conda-linux-gnu-cc
```

```bash
x86_64-conda-linux-gnu-cc --version
```

```bash
export CUDA_HOME="$CONDA_PREFIX"
```

```bash
export LIBRARY_PATH="$CUDA_HOME/lib/stubs:$CUDA_HOME/lib:${LIBRARY_PATH:-}"
```

```bash
export LD_LIBRARY_PATH="$CUDA_HOME/lib/stubs:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
```

```bash
export TCNN_CUDA_ARCHITECTURES=75
```

```bash
ls $LIBRARY_PATH | tr ':' '
'
```

```bash
echo "$LIBRARY_PATH" | tr ':' '
'
```

```bash
ls -d /home/vlab/miniforge3/envs/inr-env/lib/stubs
```

```bash
ls -d /home/vlab/miniforge3/envs/inr-env/lib
```

```bash
nvidia-smi
```

```bash
python -m pip install --no-build-isolation --no-cache-dir \
   git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

```bash
python - << 'EOF'\nimport torch\nimport tinycudann as tcnn\nprint("torch:", torch.__version__, "cuda:", torch.version.cuda)\nprint("CUDA available:", torch.cuda.is_available())\nprint("tiny-cuda-nn imported OK:", tcnn.__name__)\nEOF
```

