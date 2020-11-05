# Cuda Parallel Programming Studies

## Description
My GPU was expensive, so I have to use it.  
As I can't be angry only with games, I'm becoming upset with C++ and memory allocation again, after some good months with Python.  
Learning CUDA programming here =D

## Information
The .cu files are basically .cpp with some CUDA syntax.  
In CUDA context, Host = CPU and Device = GPU.  
Variables with a `h` in the name are host's variables. 
The ones with an `d` are device's variables.  
The CUDA headers (.h files) are on `CUDA Toolkit installation folder/include`.


## How to run
### Requirements
- A NVIDIA GPU (I have a GTX 1050 Ti)
- Install CUDA Toolkit (remember to read the toolkit installation requirements
  - [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
  - [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

### Steps to run
- Download the .cu file or `git clone` this repository
- Compile your `.cu` with `nvcc`
  - `nvcc myfile.cu`
- Run the generated file
  - Windows: `a.exe`
  - Linux (not tested): `./a.out`

## Content
|      Project       |                               Description                               |
| :----------------: | :---------------------------------------------------------------------: |
| [00](src/00/00.cu) | Basic vector adder, with CPU vs GPU comparison (a kind of hello world). |
|         01         |                              To implement                               |

