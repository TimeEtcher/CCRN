import torch.nn as nn
from . import block as B
import torch



class CRFN(nn.Module):
    def __init__(self, in_nc=3, nf=48, num_modules=3, out_nc=3, upscale=4,k=2):
        super(CRFN, self).__init__()
        self.num_modules = num_modules
        self.head = nn.Conv2d(in_nc, nf, 3, 1, 1)
        # body cells
        self.CRFBs = nn.ModuleList([B.CRFModule(in_channels=nf,kernel=k) for _ in range(num_modules)])
        # fusion
        self.local_fuse = nn.ModuleList([nn.Conv2d(nf * 2, nf, 1, 1, 0) for _ in range(num_modules-1)])
        self.GELU = nn.GELU()
        self.tail = nn.Sequential(
            nn.Conv2d(nf, out_nc * (upscale ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale)
        )

    def forward(self, input):
        # head
        out0 = self.head(input)
        x = out0
        #body cells
        out_b = [x]
        for i in range(self.num_modules):
            module = self.CRFBs[i]
            if i >= 1:
                x1 = out_b[i-1]
                x2 = out_b[i]
                x_fuse = self.local_fuse[i-1](torch.cat([x1, x2], dim=1))
                x = module(x_fuse)
            else:
                x = module(x)
            out_b.append(x)
        out_final = out_b[-1]
        out_final = self.GELU(out_final)
        out = out_final + out0
        # tail
        out = self.tail(out)

        return out