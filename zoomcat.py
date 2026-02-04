import torch
import torch.nn as nn
import torch.nn.functional as F

class Zoomcat(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # FIX: Force int
        in_channels = int(in_channels)
        reduced_c = in_channels // 2 
        
        self.conv_l = nn.Conv2d(in_channels, reduced_c, 1)
        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        #branch 2: medium size (keep original)
        self.conv_m = nn.Conv2d(in_channels, reduced_c, 1)

        #branch 3: small size (upsampling)
        self.conv_s = nn.Conv2d(in_channels, reduced_c, 1)
        

        #Final fusion
        #concatenating 3 branches of 'reduced_c' channels
        self.final_conv = nn.Conv2d(reduced_c * 3, in_channels, kernel_size=1)

    def forward(self, x):
        #x: [B, C, H, W]

        #Branch 1: large(downsample&resotre)
        #the paper says "downsampling to maintain in high resolution effectiveness"
        #Ideally, we downsample to process, then we must resize back to concat
        xl = self.conv_l(x)
        #combine max and avg pool(sum or separate? standard is usually sum or cat )
        xl_pooled = self.max_pool(xl) + self.avg_pool(xl)
        #we must resize it back to H,W to concatenate
        xl_out = F.interpolate(xl_pooled, size=x.shape[2:], mode='nearest')
        #bracnh 2: medium(identity)

        xm_out = self.conv_m(x)

        #branch 3: small(upsample)
        xs = self.conv_s(x)
        #Upsampling is performed using nearest neight 
        #effectively zooming in
        xs_up = F.interpolate(xs, size=x.shape[2:], mode='nearest')
        #Again to concatenate, we technically need matching sizes.
        #if we just upsample it's bigger
        #interpretation: the paper likely processes these scales and then brings them
        #to a common resolution for the final 'Concat'
        #Concatenate all branches
        xs_out = F.interpolate(xs, size=x.shape[2:], mode='nearest')
        
        out = torch.cat([xl_out, xm_out, xs_up], dim=1)
        out = self.final_conv(out)

        return out