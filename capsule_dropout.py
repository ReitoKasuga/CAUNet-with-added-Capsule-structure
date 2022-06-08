import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn import functional as F

class Capsule_Dropout(Module):

    __constants__ = ['p', 'inplace']
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(Capsule_Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return 'p={}, inplace={}'.format(self.p, self.inplace)

    def forward(self, input):
        device = input.device
        mask_shape = (input.shape[0], input.shape[1], 1)
        one = torch.nn.Parameter(torch.ones(mask_shape), requires_grad=False)
        one = one.to(device)

        return F.dropout(one, self.p, self.training, self.inplace)*input