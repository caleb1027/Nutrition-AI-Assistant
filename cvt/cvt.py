from modules import *

class CvT(nn.Module):
    def __init__(self, image_size, in_channels, num_classes, dim=64):
        super().__init__()

        self.dim = dim
        # Stage 1: Conv, Transformer
        trans = Transformer(dim, image_size//4, 1, 1, dim_heads=self.dim, mlp_dim=dim*4).to(device)
        self.stage_1_embed = ConvEmbedding(image_size, in_channels, 7, 4, 4, 2, self.dim).to(device)
        self.stage_1_transformer = nn.Sequential(trans,
                                                  Rearrange('b (h w) c -> b c h w', h = image_size//4, w = image_size//4)).to(device)
        # Stage 2: Conv, Transformer
        in_channels = dim
        scale = 3//1
        dim = dim * scale
        self.stage_2_embed = ConvEmbedding(image_size, in_channels, 3, 2, 8, 1, dim).to(device)
        trans = Transformer(dim, image_size//8, 2, 3, dim_heads=self.dim, mlp_dim=dim*4).to(device)
        self.stage_2_transformer = nn.Sequential(trans,
                                                  Rearrange('b (h w) c -> b c h w', h = image_size//8, w = image_size//8)).to(device)
        # Stage 3: Conv, Transformer, FFN
        in_channels = dim
        scale = 6//3
        dim = scale * dim
        self.stage_3_embed = ConvEmbedding(image_size, in_channels, 3, 2, 16, 1, dim).to(device)
        self.stage_3_transformer = nn.Sequential(Transformer(dim, image_size//16, 10, 6, dim_heads=self.dim, mlp_dim=dim*4, last_stage=True)).to(device)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)).to(device)

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        ).to(device)


    def forward(self, img):

        xs = self.stage_1_embed(img)
        xs = self.stage_1_transformer(xs)

        xs = self.stage_2_embed(xs)
        xs = self.stage_2_transformer(xs)

        xs = self.stage_3_embed(xs)
        b, n, _ = xs.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        xs = torch.cat((cls_tokens, xs), dim=1)
        xs = self.stage_3_transformer(xs)
        xs = xs[:, 0]

        xs = self.mlp(xs)
        return xs

