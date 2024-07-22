# DGTNet:Dynamic Graph Attention Transformer Network for Traffic Flow Forecasting 
![Python 3.8](https://img.shields.io/badge/Python-3.8-green.svg?style=plastic)
![PyTorch 1.8](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)

## DGTNet
<p align='center'>
  <img src="DGTNet.png" alt="architecture" width="600"/>
</p>

## Code
- `main.py` : main DGTNet model interface with training and testing
- `DGT.py` : with DGTNet class implementation
- `utils.py` : utility functions and dataset parameter
- `dataset_DGTNet.py` : generate graph from dataset

### Dataset
The traffic data files for Los Angeles (METR-LA) and the Bay Area (PEMS-BAY), i.e., `metr-la.h5` and `pems-bay.h5`, are available at [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g), and should be
put into the `data/` folder.


## Model parameters
The parameters setting can be found in `utils.py`.
- `l_backcast` : lengths of backcast
- `d_edge` : number of IMF used
- `d_model` : the time embedding dimension
- `N` : number of Self_Attention_Block
- `h` : number of head in Multi-head-attention
- `N_dense` : number of linear layer in Sequential feed forward layers
- `n_output` : number of output (lengths of forecast $\times$ number of node)
- `n_nodes` : number of node (aka number of time series)
- `lambda` : the initial value of the trainable lambda $\alpha_i$
- `dense_output_nonlinearity` the nonlinearity function in dense output layer

## Requirements
- Python 3.8
- PyTorch = 1.8.0 (with corresponding CUDA version)
- Pandas = 1.4.0
- Numpy = 1.22.2
- PyEMD = 1.2.1

Dependency can be installed using the following command:
```bash
pip install -r requirements.txt
```


## Contact
If you have any questions, please feel free to contact Chen jing (Email: 13891739600@163.com).
