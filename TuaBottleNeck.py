import torch
import torch.nn as nn
from TuaAttention import TuaAttention

class TuaBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut = True, e=0.5):
        """
        c1: input channels
        c2: output channels
        shortcut: add residual connection(True/False)
        e: expansions ratio(not strictly used in paper's formula but common in Yolo blocks).
        we will ignore'e' to stick to paper's eqns 5-7
        """

        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.add = shortcut and c1 == c2 #only add residual if channels match

        #1. layer norms
        #layernorm is usually expect (n, h, w,c ) but image data is (N, C, H, W)
        #we need a helper or use group norm which is friendlier to CNNs
        #the paper says Layer norm so strictly we should permute dimensions
        self.ln1 = nn.LayerNorm(c1)
        self.ln2 = nn.LayerNorm(c2)

        #attention module
        self.attn = TuaAttention(c1)

        #the mlp (eq 7 in paper)
        #mlp = conv1 -> gelu -> conv2

        self.mlp = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size = 1),
            nn.GELU(),
            nn.Conv2d(c2, c2, kernel_size = 1)
        )
    def forward(self, x):
        #x shape: [Batch, Channel, Height Width]
        #PART 1: Attention (EQ.5)
        residual = x

        y = x.permute(0, 2, 3, 1)
        y = self.ln1(y)
        y = y.permute(0, 3, 1, 2)
        y= self.attn(y)

        if self.add:
            y = y + residual

        residual = y

        z = y.permute(0, 2, 3, 1)
        z = self.ln2(z)
        z = z.permute(0, 3, 1, 2)

        z = self.mlp(z)

        if self.add:
            z = z + residual

        return z