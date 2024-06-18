from layers import *

from torch import nn, einsum
from torchvision import datasets
from torchvision import transforms
from einops.layers.torch import Rearrange
from einops import rearrange

class Transformer(nn.Module):
    def __init__(self, dim, img_size, depth, heads, dim_heads, mlp_dim, dropout=0., last_stage=False):
        super().__init__()
        self.modules = []
        for _ in range(depth):
            self.modules.append(nn.ModuleList([
                Norm(dim, ConvAttention(dim, img_size, heads=heads, dim_head=dim_heads, dropout=dropout, last_stage=last_stage)).to(device),
                Norm(dim, FeedForward(dim, mlp_dim, dropout=dropout)).to(device)
            ]))
    def forward(self, x):
        for attn, ff in self.modules:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ConvEmbedding(nn.Module):
    def __init__(self, image_size, in_channels, kernel_size, stride, size, padding, dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, dim, kernel_size, stride, padding),
            Rearrange('b c h w -> b (h w) c', h = image_size//size, w = image_size//size),
            nn.LayerNorm(dim)).to(device)
    def forward(self, x):
        x = self.conv(x)
        return x
    
class ConvProj(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding = 0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, padding=padding).to(device)
        self.bn = nn.BatchNorm2d(in_channels).to(device)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1).to(device)
    def forward(self, input):
        x = self.depthwise(input)
        x = self.bn(x)
        x = self.pointwise(x)
        return x
    
class ConvAttention(nn.Module):
     def __init__(self, dim, img_size, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False):

        super().__init__()
        self.last_stage = last_stage
        self.img_size = img_size
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        # map to q,k,v
        self.heads = heads
        self.scale = dim_head ** -0.5
        pad = (kernel_size - q_stride)//2
        self.to_q = ConvProj(dim, inner_dim, kernel_size, pad).to(device)
        self.to_k = ConvProj(dim, inner_dim, kernel_size, pad).to(device)
        self.to_v = ConvProj(dim, inner_dim, kernel_size, pad).to(device)

        # use if outputting, else use identity
        self.out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)).to(device) if project_out else nn.Identity().to(device)

     def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        if self.last_stage:
            cls_token = x[:, 0]
            x = x[:, 1:]
            cls_token = rearrange(cls_token.unsqueeze(1), 'b n (h d) -> b h n d', h = h)
        x = rearrange(x, 'b (l w) n -> b n l w', l=self.img_size, w=self.img_size)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        v = self.to_v(x)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(x)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)

        if self.last_stage:
            q = torch.cat((cls_token, q), dim=2)
            v = torch.cat((cls_token, v), dim=2)
            k = torch.cat((cls_token, k), dim=2)


        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.out(out)
        return out
