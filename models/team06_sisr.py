import torch
import torch.nn as nn
import torch.nn.functional as F


no_bb = 3
no_cb = 5
nf = 64

def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
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

def act(act_type, inplace=True, neg_slope=0.2, n_selu=1):
    # helper selecting activation
    # neg_slope: for selu and init of selu
    # n_selu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU()
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(0.2,inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU()
    elif act_type == 'sigmoid':
        layer = nn.Sigmoid()
    elif act_type == 'selu':
        layer = nn.SELU()
    elif act_type == 'elu':
        layer = nn.ELU()
    elif act_type == 'silu':
        layer = nn.SiLU()
    elif act_type == 'rrelu':
        layer = nn.RReLU()
    elif act_type == 'celu':
        layer = nn.CELU()
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer

def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='elu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)

class EMHA(nn.Module):
    def __init__(self, inChannels, splitfactors=4, heads=8):
        super().__init__()
        dimHead = inChannels // (2*heads)

        self.heads = heads
        self.splitfactors = splitfactors
        self.scale = dimHead ** -0.5

        self.reduction = nn.Conv1d(
            in_channels=inChannels, out_channels=inChannels//2, kernel_size=1)
        self.attend = nn.Softmax(dim=-1)
        self.toQKV = nn.Linear(
            inChannels // 2, inChannels // 2 * 3, bias=False)
        self.expansion = nn.Conv1d(
            in_channels=inChannels//2, out_channels=inChannels, kernel_size=1)

    def forward(self, x):
        x = self.reduction(x)
        x = x.transpose(-1, -2)

        qkv = self.toQKV(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        qs, ks, vs = map(lambda t: t.chunk(
            self.splitfactors, dim=2), [q, k, v])

        pool = []
        for qi, ki, vi in zip(qs, ks, vs):
            tmp = torch.matmul(qi, ki.transpose(-1, -2)) * self.scale
            attn = self.attend(tmp)
            out = torch.matmul(attn, vi)
            out = rearrange(out, 'b h n d -> b n (h d)')
            pool.append(out)

        out = torch.cat(tuple(pool), dim=1)
        out = out.transpose(-1, -2)
        out = self.expansion(out)
        return out

class CAM(nn.Module):
	def __init__(self, in_channels, reduction_ratio=16):
		super().__init__()
		self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
		self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)

	def forward(self, x):
		avg_pool = self.global_avg_pool(x)
		avg_pool = avg_pool.view(avg_pool.size(0), -1)
		fc1_output = F.relu(self.fc1(avg_pool))
		channel_attention = torch.sigmoid(self.fc2(fc1_output))
		x = x * channel_attention.unsqueeze(2).unsqueeze(3)
		return x

class BB (nn.Module) :
  def __init__(self, nf, splitfactors=4, heads=8, k=3) :
    super(BB, self).__init__()
    self.k = k
    self.uk3_1 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 3)
    self.uk3_2 = conv_block(in_nc=2*nf, out_nc=nf, kernel_size= 3)
    self.uk3_3 = conv_block(in_nc=3*nf, out_nc=nf, kernel_size= 3)
    self.lk3_1 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 3)
    self.lk3_2 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 3)
    self.lk3_3 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 3)
    self.k1 = conv_block(in_nc=4*nf, out_nc=nf, kernel_size= 1)
    self.emha = EMHA(nf*k*k, splitfactors, heads)
    self.norm = nn.LayerNorm(nf*k*k)
    self.unFold = nn.Unfold(kernel_size=(k, k), padding=1)
  
  def forward(self,x):
    _, _, h, w = x.shape

    #upper path
    xu1_1= self.uk3_1(x)
    xu1_2= torch.cat((xu1_1,x),1)
    xu2_1= self.uk3_2(xu1_2)
    xu2_2= torch.cat((xu2_1,xu1_1,x),1)
    xu3_1= self.uk3_3(xu2_2)
    xu3_2= torch.cat((xu3_1,xu2_1,xu1_1,x),1)
    xu3= self.k1(xu3_2)
    #lower path
    xl1= self.lk3_1(x)
    xl1= self.lk3_2(xl1)
    xl1= self.lk3_3(xl1)
    xl1= x+xl1
    #transformer 

    # xt1 = self.unFold(x)
    # xt2 = xt1.transpose(-2, -1)
    # xt2 = self.norm(xt2)
    # xt2 = xt2.transpose(-2, -1)
    # xt2 = self.emha(xt2)+xt1
    # xt2 = F.fold(xt2, output_size=(h, w), kernel_size=(self.k, self.k), padding=(1, 1))
    # xt2 = xt2+x

    return xu3+xl1+x

class CB (nn.Module) :
  def __init__(self, nf, no_bb) :
    super(CB, self).__init__()
    self.fwd_bb = nn.ModuleList([BB(nf) for i in range(no_bb)])
    self.fwd_cam = nn.ModuleList([CAM(nf) for i in range (no_bb)])
    self.no_bb = no_bb

  def forward(self,x) :
    x1 = self.fwd_bb[0](x)
    x1 = self.fwd_cam[0](x1)
    for i in range(self.no_bb-1):
      x1 = self.fwd_bb[i+1](x1)
      x1 = self.fwd_cam[i+1](x1)    
    return x1 + x

class OurUpSample(nn.Module):
    def __init__(self,in_nc, kernel_size=3, stride=1, bias=True, pad_type='zero', \
            act_type=None, mode='CNA',upscale_factor=2):
        super(OurUpSample, self).__init__()
        self.U1 = pixelshuffle_block(in_nc, in_nc, upscale_factor=upscale_factor, kernel_size=3, norm_type = 'batch')
        self.co1 = conv_block(in_nc, in_nc, kernel_size=1, norm_type=None, act_type='elu', mode='CNA')
        # self.co2 = conv_block(16, 3, kernel_size=3, norm_type=None, act_type='prelu', mode='CNA')

    def forward(self, x):
        out1 = self.U1(x)
        return self.co1(out1)

def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                        pad_type='zero', norm_type=None, act_type='relu'):
    '''
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    '''
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=None, act_type=None)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)

class MyNetwork (nn.Module) :
  def __init__(self, nf=nf, no_cb=no_cb, no_bb=no_bb, in_c = 3) :
    super(MyNetwork, self).__init__()
    self.k5 = conv_block(in_nc=in_c, out_nc=nf, kernel_size= 5)
    self.cb = CB(nf, no_bb)
    self.fwd_cb = nn.ModuleList([CB(nf, no_bb) for i in range (no_cb - 1)])
    self.ub = nn.Upsample(scale_factor=4, mode='nearest')
    # self.ub = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
    #                         nn.Upsample(scale_factor=4, mode='nearest'))
    self.u = OurUpSample(nf, kernel_size=3, act_type='elu',upscale_factor=4)
    # self.up1 = pixelshuffle_block(nf, nf, upscale_factor=2,norm_type = 'batch')
    # self.up2 = pixelshuffle_block(nf, nf, upscale_factor=2,norm_type = 'batch')
    #self.conv2 = conv_block(nf,nf,kernel_size=kernel_size,norm_type=norm_type,act_type=act_type)
    #self.conv3 = conv_block(nf,out_nc,kernel_size=kernel_size,norm_type=norm_type,act_type=act_type)
    self.k3 = conv_block(in_nc=nf, out_nc=nf, kernel_size= 3)
    self.k3_1 = conv_block(in_nc=nf, out_nc=in_c, kernel_size= 3)

  def forward (self, x) :
    #upper 
    xu1 = self.k5(x)
    xu2 = self.cb(xu1)
    for l in self.fwd_cb :
      xu2 = l(xu2)
    xu2 = xu2 + xu1
    xu2 = self.u(xu2)
    #xu2 = self.up2(xu2)
    xu2 = self.k3(xu2)
    xu2= self.k3_1(xu2)
    #lower
    xl1= self.ub(x)

    return xl1+xu2


class ProposedNetwork(nn.Module):
    def __init__(self, nf=nf, no_cb=no_cb, no_bb=no_bb):
        super(ProposedNetwork, self).__init__()
        self.model = MyNetwork(nf, no_cb, no_bb)

    def forward(self, x):
        x = self.model(x)
        return x