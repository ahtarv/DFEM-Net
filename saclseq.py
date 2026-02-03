import torch
import torch.nn as nn
import torch.nn.functional as F

class Scalesq(nn.Module):
    def __init__(self, c_p3, c_p4, c_p5, out_channels =256):
        """
        c_p3, c_p4, c_p5: Input channels for the three feature maps
        output_chanels: the fixed channel count for fusion(paper uses 256)
        """
        super().__init__()
        self.conv_p3 = nn.Conv2d(c_p3, out_channels, kernel_size=1)
        self.conv_p4 = nn.Conv2d(c_p4, out_channels, kernel_size=1)
        self.conv_p5 = nn.Conv2d(c_p5, out_channels, kernel_size=1)

        #the 3d convolution
        #input: (bathc, channel, depth = 3, height, width)
        #kernel: (3,1,1) means we mix across all 3 scales(depth=3),
        #but look at 1x1 pixels spaitally
        self.conv3d = nn.Conv3d(
            in_channels = out_channels,
            out_channels=out_channels,
            kernel_size=(3,1,1),
            padding=0,
            bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.SiLU() #paper mentions silu

    def forward(self, p3, p4, p5):
        #p3: [B, C3, H, w]
        #p4 [B, c4, h/2, w/2]
        #p5: [b, c5, h/4, w/4]

        f3 = self.conv_p3(p3)
        f4 = self.conv_p4(p4)
        f5 = self.conv_p5(p5)

        target_h, targeta_w = f3.shape[2], f3.shape[3]
        f4 = F.interpolate(f4, size=(target_h, target_w), mode = 'nearest')
        f5 = F.interpolate(f5, size=(target_h, target_w), mode ='nearest')

        f_stacked = torch.stack([f3, f4, f4], dim=2)

        y = self.conv3d(f_stacked)
        y = self.bn(y)
        y = self.act(y)

        return y.squeeze(2)