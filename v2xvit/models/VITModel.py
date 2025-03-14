import torch.nn as nn

#第一步：patch_embedding模块 将图片切割成16*16
class patchEmbedding(nn.Module):
    """
    2D Image to patch embedding 
    """
    def __init__(self, img_size=224, patch_size=16, in_c = 3 ,embed_dim =768,norm_layer=None):
        super(patchEmbedding,self).__init__()
        img_size = (img_size,img_size)#输入图片大小(224*224)
        patch_size = (patch_size,patch_size)#卷积核大小(16*16)
        self.img_size =img_size
        self.proj = nn.Conv2d(in_c,embed_im,kernel_size=patch_size,stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B,C,H,W = x.shape
        x = self.proj(x).flatten(2).transpose(1,2)
        x = self.norm(x)
        return x

class MultiAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qk_scale=None,
                 qkv_bias = False
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(MultiAttention,self).__init__()
        self.num_heads = num_heads
        head_dim = dim//num_heads
        self.scale= qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim,dim*3,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)


    def forward(self,x)
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2]
        attn = (q @ k.transpose(-2,-1))*self.scale
        attn = attn.softmax(dim=1)
        attn = self.attn_drop(attn)

        x = (attn @ V).transpose(1,2).reshape(B,N,C)
        x = self.proj(x)#拼接后进行W0映射
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None,out_feature= None, act_layer=nn.GELU, drop = 0.):
        super().__init__()
        out_feature = out_feature or in_features
        hidden_features = hidden_features
        self.fc1 = nn.Linear(in_feature,hidden_features)
        self.GELU = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.Dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.GELU(x)
        x = self.Dropout(x)
        x = self.fc2(x)
        x = self.Dropout(x)
        return x

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layers=nn.GELU,
                 norm_layer=nn.LayerNorm):
    super(block,self).__init__()
    self.norm1=norm_layer(dim)
    self.attn = Multiattention(dim,num_heads=num_heads, qkv_bias=qkv_bias, qk_scale= qk_scale,
                               attn_drop_ratio=attn_drop_ratio,proj_drop_ratio=drop_ratio)
    self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio>0. else nn.Identity()
    self.norm2 = norm_layer(dim)
    mlp_hidden_dim = int(dim * mlp_ratio)
    self.mlp=MLP(in_features=dim, hidden_features=mlp_hidden_dim,, act_layer=act_layers, drop = drop_path_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224,patch_size=16,in_c=3,num_class=1000,embed_dim=768,
                 depth=12,num_heads=12,mlp_ratio=4.0,qkv_bias=True,qk_scale=None,
                 representation_size=None,distilled=False,drop_ratio=0.,attn_drop_ratio=0.1,
                 drop_path_ratio=0.1,embed_layer=patchEmbedding,norm_layer=None,act_layer=None)
    super(VisionTransformer,self).__init__()
    self.num_class = num_classed#分类头的数量。
    self.num_features = self.embed_dim = embed_dim
    self.num_token = 2 if distilled else 1 #distilled = false 兼容DeiT模型，所以num_token=1
    norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
    act_layer = act or nn.GRLU #act_layer==nn.GELU

    self.patch_embed = embed_layer(img_size=img_size, patch_size = patch_size,
                                   in_c=in_c, embed_dim = embed_dim)
    num_patches = self.patch_embed.patch_size

    self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
    # nn.Parameter 构建一个可学习参数初始化为0，第一个1 表示Batchsize维度，第二个和第三个(1,embed_dim)是
    # 1*768,要concat拼接在一起
    self.dist_token = nn.Parameter(torch.zeros(1,1,embed_dim)) if distilled else None

    #第一个1代表Batch_size维度，方便拼接，num_patches + self.num_tokens表示196+1   (B,197,768)

    self.pos_embed = nn.Parameter(torch.zeros(1,num_patches + self.num_token,embed_num))
    self.pos_drop = nn.Dropout(p=drop_ratio)


    #torch.Linspace 建立0-drop_path_ratio的等差数列，每一层的dropout的比例都是在增加的
    dpr =[x.item() for x in torch.Linspace(0, drop_path_ratio,depth)]

    #产生重复堆叠的Block
    self.blocks = nn.Sequential(*[
        Block(dim=embed_dim,num_heads = num_heads, mlp_ratio = mlp_ratio,
                 qkv_bias=qkv_bias,
                 qk_scale=qk_scale,
                 drop_ratio=drop_ratio,
                 attn_drop_ratio=attn_drop_ratio,
                 drop_path_ratio=dpr[i],
                 act_layers=act_layer,
                 norm_layer=norm_layer)
        for i in range(depth) # 创建一个列表，总共循环depth次，每次的drop_patch_ratio都不一样，
    ])
    self.norm = norm_layer(embed_dim)

    self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    #是某种初始化方法，将神经网络的参数按照某种算法的初始化，标准差为0.2。
    nn.init.trunc_normal_(self.pos_embed,std = 0.2)
    if self.dist_token is not None:
        #同样的算法，也给dist_token 初始化一下。
        nn.init.trunc_normal_(self.dist_token, std = 0.2)

    nn.init.trunc_normal_(self.cls_token, std=0.2)
    self.apply(_init_vit_weights)

    def forward_features(self, x):
        x =patchEmbedding(x)#(B,196,768)
        #[1,1,768]->[B,1,768] 根据输入的x的Batch_size，在batch_size这个维度复制batch
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x),dim = 1)#[B,196,768]->[B,197,768]        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0],-1,-1),x),dim = 1)
        zrngjia
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:,0])
        else:
            return x[:,0], x[:,1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]),self.head_dist([x[1]])

