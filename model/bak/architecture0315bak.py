import torch.nn as nn
from . import block0315bak as B
import torch
from torch.nn import functional as F


class CRFN(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_modules=3, out_nc=3, upscale=4,k=2):
        super(CRFN, self).__init__()
        self.num_modules = num_modules
        self.fea_conv = B.conv_layer(in_nc * 4, nf, kernel_size=3)
        # IMDBs
        CRFBs = [B.CRFModule(in_channels=nf,kernel=k) for _ in range(num_modules)]
        self.CRFBs = nn.ModuleList(CRFBs)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.tail = nn.Sequential(
            nn.Conv2d(nf, out_nc * (upscale ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale)
        )

    def forward(self, input):
        input = torch.cat([input, input, input, input], dim=1)
        out_fea = self.fea_conv(input)
        x = out_fea
        out_b = list()
        for i in range(self.num_modules):
            x = self.CRFBs[i](x)
            out_b.append(x)

        out_B = self.c(torch.cat([x for x in out_b], dim=1))
        out = out_B + out_fea
        output = self.tail(out)
        return output