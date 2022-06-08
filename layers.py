import numpy as np
import torch
import torch.nn as nn

class Squash(nn.Module):
    def __init__(self, eps=10e-21):
        super().__init__()
        self.eps = eps

    def forward(self, s):
        n = torch.norm(s, dim=2, keepdim=True)
        return (1 - 1 / (torch.exp(n) + self.eps)) * (s / (n + self.eps))

class PrimaryCaps(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, caps_num, caps_dim, stride=1):
        super(PrimaryCaps, self).__init__()
        self.I = in_ch
        self.F = out_ch   # F=N*D
        self.K = kernel
        self.N = caps_num
        self.D = caps_dim
        self.s = stride

        self.DW_Conv2D = nn.Conv2d(self.I, self.F, self.K, self.s,
                                   padding=[self.K//2], groups=self.F)
        self.Squash = Squash()

    def forward(self, inputs):
        x = self.DW_Conv2D(inputs)  # shape : (batch_size, C, frame_num, frame_size)
        x_shape = x.size()
        batch_size = x_shape[0]
        frame_num = x_shape[2]
        frame_size = x_shape[3]
        x = x.view(batch_size, -1, self.D, frame_num, frame_size) # shape : (batch_size, C/caps_dim, caps_dim, frame_num, frame_size)
        x = torch.permute(x, (0, 1, 4, 2, 3)) # shape : (batch_size, C/caps_dim, frame_size, caps_dim, frame_num)
        x = x.contiguous().view(batch_size, -1, self.D, frame_num) # shape : (batch_size, C/caps_dim * frame_size, caps_dim, frame_num)
        x = self.Squash(x)

        return x