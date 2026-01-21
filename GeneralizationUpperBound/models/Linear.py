import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.In = nn.Linear(self.seq_len, self.seq_len*2)
        self.act = nn.GELU()
        self.Linear = nn.Linear(self.seq_len*2, self.pred_len)
        self.adnorm = AdaptiveMovingNormalize(seq_len=self.seq_len, window_size=50, use_bias=True)
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = self.Linear(
                self.act(
                    self.In(
                        self.adnorm.norm(x).permute(0,2,1)
                    )
                )
            ).permute(0,2,1)
        return self.adnorm.denorm(x) # [Batch, Output length, Channel]

class AdaptiveMovingNormalize(nn.Module):
    def __init__(self, seq_len: int, window_size: int, eps: float = 1e-2, use_bias: bool = True):
        super().__init__()
        self.seq_len = seq_len
        self.window_size = window_size
        self.eps = eps
        self.use_bias = use_bias

        self.scale = torch.nn.Parameter(torch.ones(seq_len,1))
        self.descale = torch.nn.Parameter(torch.ones(seq_len,1))

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(seq_len,1)) 
            self.debias = torch.nn.Parameter(torch.zeros(seq_len,1))
        else:
            self.bias = None
            self.debias = None

        self.mean = None
        self.std = None

    def norm(self, x:torch.Tensor) -> torch.Tensor:
        """
        x will be normalized along dim in rolling windows,
        - x: (dim >= 3) validation shape is [..., seq_len, :]
        - dim: (tuple of int) the dimension to be normalized, default is (-1,)
        """

        if self.use_bias:
            ma, mstd = self._moving_std_mean_unfold(x)
            self.mean = ma
            self.std = mstd
            return (x - ma) / (mstd + self.eps) * self.scale + self.bias
        else:
            _, mstd = self._moving_std_mean_unfold(x)
            self.std = mstd
            return x / (mstd + self.eps) * self.scale 
    
    def denorm(self, x:torch.Tensor) -> torch.Tensor:
        if self.use_bias:
            return (x * self.descale + self.debias) * self.std + self.mean
        else:
            return x * self.descale * self.std

    def _moving_std_mean_unfold(self, x:torch.Tensor) -> torch.Tensor:
        """
        create the moving mean and std
        - x: validation shape is [..., seq_len, :]
        """
        ma = torch.zeros(size=(1,1,1)) # 占位符，避免编译报错
        if self.use_bias:
            ma = torch.zeros_like(x) # 重新定义
        mstd = torch.ones_like(x)
        for i in range(self.seq_len-self.window_size):
            if self.use_bias:
                ma[..., i+self.window_size:i+self.window_size+1,:] = torch.mean(x[..., i:i+self.window_size, :], dim=-2, keepdim=True)
            mstd[..., i+self.window_size:i+self.window_size+1,:] = torch.std(x[..., i:i+self.window_size, :], dim=-2, keepdim=True)
        for i in range(self.window_size):
            if self.use_bias:
                ma[..., i:i+1,:] = torch.mean(x[..., :i+1, :], dim=-2, keepdim=True)
            if i > 0:
                mstd[..., i:i+1,:] = mstd[..., self.window_size:self.window_size+1,:]

        return ma, mstd