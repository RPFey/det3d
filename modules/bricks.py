from torch import nn as nn
import torch

NORM_LAYER = {
    'BN2D': nn.BatchNorm2d
}

ACT_LAYER = {
    'ReLU': nn.ReLU
}

class MLP(nn.Module):
    def __init__(self, in_features, out_features, norm_cfg = dict(type='BN2D'), act_cfg = dict(type='ReLU')):
        super().__init__()
        assert norm_cfg is None or norm_cfg['type'] in NORM_LAYER.keys(), "norm layer is not supported"
        assert act_cfg is None or act_cfg['type'] in ACT_LAYER.keys(), "act layer not supported"
        self.block = nn.Sequential()
        self.block.add_module('conv', nn.Conv2d(in_features, out_features, kernel_size=1))
        if norm_cfg is not None:
            self.block.add_module('norm', NORM_LAYER[norm_cfg['type']](out_features))
        if act_cfg is not None:
            self.block.add_module('act', ACT_LAYER[act_cfg['type']]())
    
    def forward(self, x):
        return self.block(x)