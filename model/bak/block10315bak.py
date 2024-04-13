import torch.nn as nn
from collections import OrderedDict
import torch
from torch.nn import functional as F


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                     groups=groups)


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == 'gelu':
        layer = nn.GELU()
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

# 输入为 [N, C, H, W]，需要两个参数，in_planes为输特征通道数，K 为专家个数
class Attention(nn.Module):
    def __init__(self,in_planes,K):
        super().__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.net=nn.Conv2d(in_planes, K, kernel_size=1)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        # 将输入特征全局池化为 [N, C, 1, 1]
        att=self.avgpool(x)
        # 使用1X1卷积，转化为 [N, K, 1, 1]
        att=self.net(att)
        # 将特征转化为二维 [N, K]
        att=att.view(x.shape[0],-1)
        # 使用 sigmoid 函数输出归一化到 [0,1] 区间
        return self.sigmoid(att)

class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class CondConv(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,stride,padding=0,
                 groups=1,K=2):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.K = K
        self.groups = groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.attention = Attention(in_planes=in_planes,K=K)
        self.weight = nn.Parameter(torch.randn(K,out_planes,in_planes//groups,
                                             kernel_size,kernel_size),requires_grad=True)

    def forward(self,x):
        # 调用 attention 函数得到归一化的权重 [N, K]
        N,in_planels, H, W = x.shape
        softmax_att=self.attention(x)
        # 把输入特征由 [N, C_in, H, W] 转化为 [1, N*C_in, H, W]
        #x=x.view(1, -1, H, W)
        x = x.contiguous().view(1, -1, H, W)

        # 生成随机 weight [K, C_out, C_in/groups, 3, 3] (卷积核一般为3*3)
        # 注意添加了 requires_grad=True，这样里面的参数是可以优化的
        weight = self.weight
        # 改变 weight 形状为 [K, C_out*(C_in/groups)*3*3]
        weight = weight.view(self.K, -1)

        # 矩阵相乘：[N, K] X [K, C_out*(C_in/groups)*3*3] = [N, C_out*(C_in/groups)*3*3]
        aggregate_weight = torch.mm(softmax_att,weight)
        # 改变形状为：[N*C_out, C_in/groups, 3, 3]，即新的卷积核权重
        aggregate_weight = aggregate_weight.view(
            N*self.out_planes, self.in_planes//self.groups,
            self.kernel_size, self.kernel_size)
        # 用新生成的卷积核进行卷积，输出为 [1, N*C_out, H, W]
        output=F.conv2d(x,weight=aggregate_weight,
                        stride=self.stride, padding=self.padding,
                        groups=self.groups*N)
        # 形状恢复为 [N, C_out, H, W]
        output=output.view(N, self.out_planes, H, W)
        return output

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

# contrast-aware channel attention module
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class CRFModule(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25,kernel=2):
        super(CRFModule, self).__init__()
        self.c1 = CondConv(in_channels,in_channels,3,1,padding=1,K=kernel)
        self.c2 = CondConv(in_channels, in_channels, 3,1,padding=1,K=kernel)
        self.c3 = CondConv(in_channels, in_channels, 3,1,padding=1,K=kernel)

        self.fuse = nn.Conv2d(in_channels * 2, in_channels, 1, 1, 0)
        self.att = ESA(in_channels, in_channels, nn.Conv2d)
        self.cca = CCALayer(in_channels)
        self.branch = nn.ModuleList([nn.Conv2d(in_channels, in_channels // 2, 1, 1, 0) for _ in range(4)])
    def forward(self, input):
        out1 = self.c1(input)
        out2 = self.c2(out1)
        out3 = self.c3(out2)

        # fuse [x, out1, out2, out3]
        out = self.fuse(
            torch.cat([self.branch[0](input), self.branch[1](out1), self.branch[2](out2), self.branch[3](out3)], dim=1))
        out = self.cca(out)
        out = self.att(out)
        out += input
        return out