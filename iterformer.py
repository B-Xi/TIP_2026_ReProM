import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from mamba_ssm import Mamba
def continues_scan(input):
    # t,h,w = input.shape
    # input = input.pad((0,1,0,1,0,1), mode='constant')
    # t_,h_,w_ = input.shape
    #print(input.shape)
    input_rev = input.flip(-1)#双向
    #print(input_rev.shape)
    input_ = input.clone()
    input_[:,:,:,1::2,:] = 0
    input_rev[:,:,:,0::2,:] = 0
    input_ = torch.add(input_, input_rev)
    # input_rev = input.flip(-2).contiguous()
    # input_rev[:,:,1:2] = 0
    # input[:,:,0:1] = 0
    # input = input+input_rev
    input_rev_ = input_.flip(-2).flip(-1)
    input_rev_[:,:,0::2,:,:] = 0
    input_[:,:,1::2,:,:] = 0
    # output = input+input_rev
    output = torch.add(input_, input_rev_)
    return output

def rev_continues_scan(input):
    input_rev = input.flip(-1).flip(-2)
    input_ = input.clone()
    input_rev[:,:,0::2,:,:] = 0
    input_[:,:,1::2,:,:] = 0
    input_ = torch.add(input_, input_rev)
    input_rev_ = input_.flip(-1)
    input_[:,:,:,1::2,:] = 0
    input_rev_[:,:,:,0::2,:] = 0
    output = torch.add(input_, input_rev_)
    return output

class EMA1(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA1, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
       
        self.conv3x3 = nn.Sequential(nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU())
    
    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w


        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        #print(group_x.shape,x_h.shape,x_w.shape,(group_x * x_h.sigmoid()).shape)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        ###
        x_hw=self.agp(x2)
        ###
        x11 = self.softmax((self.agp(x1)+x_hw).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        ###
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        #print(x11.shape,x12.shape)
        weights = (torch.matmul(x11, x12) ).reshape(b * self.groups, 1, h, w)
        #print(weights.shape,group_x.shape)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.1):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.nn1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.layernorm(x)
        x = self.nn1(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.nn2(x)
        x = self.drop(x)
        return x
    

class MAA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_memory = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, memories):
        x = self.norm(x)
        x_kv = x 
    
        q, k, v = (self.to_q(x), *self.to_kv(x_kv).chunk(2, dim = -1))
        #memories = self.to_memory(memories)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))
        #memories = rearrange(memories, 'b n (h d) -> b h n d', h = self.heads)
        #print(k.shape,v.shape)
        k = k#torch.cat((k, memories), dim = 2)
        v = v#torch.cat((v, memories), dim = 2)
        #print(k.shape,v.shape)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class SA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_memory = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        x_kv = x 

        q, k, v = (self.to_q(x), *self.to_kv(x_kv).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))
        
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MAA(dim, heads, dim_head, dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]))

    def forward(self, x, memories):
        for _, (attn, ff) in enumerate(self.layers):
            x1=x
            x = attn(x, memories = memories)+x
        
            x = ff(x) + x

        return x

class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = round(dim // n_div)
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, channel_token = False,SSAS=None):
        super(MambaLayer,self).__init__()
        #print(f"MambaLayer: dim: {dim}")
        self.SSAS = SSAS
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=32, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand    # Block expansion factor
        )
        self.mamba_b=Mamba(
                d_model=32, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand    # Block expansion factor
        )
        self.channel_token = channel_token ## whether to use channel as tokens

    def forward(self, x, r = False):
        B, C, H, W = x.shape
        x=x.reshape(B,C//1,1,H,W)
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> ' + self.SSAS, b=B, c=C, t=T, h=H, w=W)
        #print(x.shape)
        x = continues_scan(x)
        #print(x.shape)
        B, d_model = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).contiguous().transpose(-1, -2)
        #print(x_flat.shape)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        x_mamba_b = self.mamba_b(x_norm.flip(1)).flip(1)
        x_mamba = x_mamba + x_mamba_b
        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)
        out = rev_continues_scan(out).contiguous()
        out = rearrange(out , self.SSAS + ' -> b c t h w', b=B, c=C, t=T, h=H, w=W)
        out=out.reshape(B,-1,H,W)
        return out
    
class GMambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, channel_token = False,SSAS=None):
        super(GMambaLayer,self).__init__()
        #print(f"MambaLayer: dim: {dim}")
        self.SSAS = SSAS
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand    # Block expansion factor
        )
        self.mamba_b=Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand    # Block expansion factor
        )
        self.channel_token = channel_token ## whether to use channel as tokens

    def forward(self, x, r = False):
        B, C, H, W = x.shape
        x=x.reshape(B,C//1,1,H,W)
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> ' + self.SSAS, b=B, c=C, t=T, h=H, w=W)
        #print(x.shape)
        x = continues_scan(x)
        #print(x.shape)
        B, d_model = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).contiguous().transpose(-1, -2)
        #print(x_flat.shape)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        x_mamba_b = self.mamba_b(x_norm.flip(1)).flip(1)
        x_mamba = x_mamba + x_mamba_b
        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)
        out = rev_continues_scan(out).contiguous()
        out = rearrange(out , self.SSAS + ' -> b c t h w', b=B, c=C, t=T, h=H, w=W)
        out=out.reshape(B,-1,H,W)
        return out
   
class massformer(nn.Module):
    def __init__(
        self,
        *,
        num_classes = 0,
        num_tokens = 0,
        dim = 64,
        depth = 3,
        heads = 8,
        dim_head = 8,
        mlp_dim = 512,
        dropout = 0.2,
        emb_dropout = 0.1,
    ):
        super().__init__()

        #self.EMA=EMA1(channels=64,factor=16)
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 3)),
            nn.ReLU(),
        )

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=784, out_channels=64, kernel_size=(1, 1)),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten(1, 2)

        self.nn = nn.Linear(num_tokens, dim)
        torch.nn.init.xavier_uniform_(self.nn.weight)
        self.dropout = nn.Dropout(dropout)

        self.pos_embedding = nn.Parameter(torch.empty(1, num_tokens, dim//2))
        torch.nn.init.normal_(self.pos_embedding, std=.001)
        
        self.transformer = Transformer(dim//2, depth, heads, dim_head, mlp_dim, emb_dropout)

        self.drop = nn.Dropout(emb_dropout)

        self.maxpool = nn.MaxPool2d(kernel_size=(7, 1), stride=(5, 1))
        self.avgpool = nn.AvgPool2d(kernel_size=(3, 1), stride=(3, 1))


        self.conv1x1 =nn.Conv2d(dim, dim,kernel_size=1)
        self.to_out = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(128)
        )
        self.conv1x12=nn.Sequential(
            nn.Conv2d(dim, 128,kernel_size=1),
            nn.ReLU(),
        )

        self.mambalayer=MambaLayer(dim = 32,  SSAS='b c t h w')
        self.mambalayer1=GMambaLayer(dim = 8,  SSAS='b c t h w')

    def forward(self, x):
        #print(x.shape)
        x=x.unsqueeze(1)
        #print(xcenter.shape)
        x = self.conv3d(x)
        #print(x.shape)
        #print(x.shape)
        x = self.flatten(x)
        #print(x.shape)
        #x=self.pConv(x)
        #print(x.shape)
        x = self.conv2d(x)
        #print(x.shape)
        x1=x
        #x=self.EMA(x)
        xconv=self.conv1x1(x1)
        #print(xconv.shape)
        xconv1, xconv2 = torch.split(xconv, [xconv.shape[1]//2, xconv.shape[1]//2], dim=1)#分开
        #print(xconv1.shape)
        xconv11=xconv1
        xconv1_1,xconv1_2,xconv1_3,xconv1_4=torch.split(xconv1, [xconv1.shape[1]//4, xconv1.shape[1]//4,xconv1.shape[1]//4, xconv1.shape[1]//4], dim=1)#分开
        xconv1=torch.cat((self.mambalayer1(xconv1_1),self.mambalayer1(xconv1_2),self.mambalayer1(xconv1_3),self.mambalayer1(xconv1_4)), dim=1)+xconv1
        #xconv1_1,xconv1_2,xconv1_3,xconv1_4,xconv1_5,xconv1_6,xconv1_7,xconv1_8,xconv1_9,xconv1_10,xconv1_11,xconv1_12,xconv1_13,xconv1_14,xconv1_15,xconv1_16=torch.split(xconv1, [xconv1.shape[1]//16, xconv1.shape[1]//16,xconv1.shape[1]//16, xconv1.shape[1]//16,xconv1.shape[1]//16, xconv1.shape[1]//16,xconv1.shape[1]//16, xconv1.shape[1]//16,xconv1.shape[1]//16, xconv1.shape[1]//16,xconv1.shape[1]//16, xconv1.shape[1]//16,xconv1.shape[1]//16, xconv1.shape[1]//16,xconv1.shape[1]//16, xconv1.shape[1]//16], dim=1)#分开
        #xconv1=torch.cat((self.mambalayer1(xconv1_1),self.mambalayer1(xconv1_2),self.mambalayer1(xconv1_3),self.mambalayer1(xconv1_4),self.mambalayer1(xconv1_5),self.mambalayer1(xconv1_6),self.mambalayer1(xconv1_7),self.mambalayer1(xconv1_8),self.mambalayer1(xconv1_9),self.mambalayer1(xconv1_10),self.mambalayer1(xconv1_11),self.mambalayer1(xconv1_12),self.mambalayer1(xconv1_13),self.mambalayer1(xconv1_14),self.mambalayer1(xconv1_15),self.mambalayer1(xconv1_16)), dim=1)+xconv1        
        
        #print(xconv2.shape)
        xconv2=self.mambalayer(xconv2)
        xconv2 = rearrange(xconv2,'b c h w -> b (h w) c')   
        xconv21=xconv2.clone()     
        #print(x.shape)
        xconv11=rearrange(xconv11,'b c h w -> b (h w) c')  
        max_token= self.maxpool(xconv11)
        #print(max_token.shape)
        avg_token = self.avgpool(xconv11)
        #print(avg_token.shape)
        memories = torch.cat((avg_token, max_token), dim=1)
        #print(memories.shape)
        memories = self.drop(memories)   
        #print(memories.shape)
        #print(cls_token.shape)
        xconv21 += self.pos_embedding
        #print(xconv21.shape)
        xconv21 = self.dropout(xconv21)
        xconv21 = self.transformer(xconv21,memories)
        xconv21 = rearrange(xconv21, 'b (h w) c -> b c h w', h=7, w=7)
        
        x = torch.cat((xconv1, xconv21), dim=1)+x
        x=self.conv1x12(x)
        token = self.to_out(x)
        #x = self.mlp_head(token)

        return token


if __name__ == '__main__':
    inputs = torch.randn(8, 100, 9, 9).cuda(1)
    features = massformer(num_classes=9, num_tokens=49).cuda(1)(inputs)
    print(features.size())