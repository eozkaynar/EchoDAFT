import torch

class Identity(torch.nn.Module):
    '''
    Idle function passing value as they are.
    '''
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Conv2Plus1D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, stride, padding):
        super(Conv2Plus1D, self).__init__()

        self.spatial    = torch.nn.Conv3d(in_channels, mid_channels, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, padding, padding), bias=False)
        self.bn3d_1     = torch.nn.BatchNorm3d(mid_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu       = torch.nn.ReLU(inplace=True)
        self.temporal   = torch.nn.Conv3d(mid_channels, out_channels, kernel_size=(3, 1, 1), stride=(stride, 1, 1), padding=(padding, 0, 0), bias=False)
        self.bn3d_2     = torch.nn.BatchNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, x):
        x = self.spatial(x)
        x = self.bn3d_1(x)
        x = self.relu(x)
        x = self.temporal(x)
        x = self.bn3d_2(x)
        return x
    
class BasicBlock2Plus1D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding, downsample=False):
        super(BasicBlock2Plus1D, self).__init__()

        
        mid_channels    = (in_channels * out_channels * 3 * 3 * 3) // (in_channels * 3 * 3 + out_channels * 3)

        self.conv1  = Conv2Plus1D(in_channels, out_channels, mid_channels=mid_channels, stride=stride, padding=padding)
        self.conv2  = Conv2Plus1D(out_channels, out_channels, mid_channels=mid_channels, stride=1, padding=padding)
        self.relu   = torch.nn.ReLU(inplace=True)
        self.downsampleconv = None
        if downsample:
            self.downsampleconv = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=(stride, stride, stride), bias=False),
                torch.nn.BatchNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsampleconv is not None:
            identity = self.downsampleconv(identity)
        out += identity
        out = self.relu(out)
        return out

class DAFTBlk(torch.nn.Module):
    def __init__(self, in_channels, out_channels, tabular_dim):
        super(DAFTBlk, self).__init__()
        # DAFT bottleneck coefficient (value = 7) as in the original study
        bottleneck_dim  = (in_channels + tabular_dim) // 7
        self.split_size = in_channels 

        # DAFT blk
        # Affine Transform   
        # Global avg pooling
        self.global_avg_pool = torch.nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        layers = [
            torch.nn.Linear(in_channels + tabular_dim, bottleneck_dim, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(bottleneck_dim, 2*self.split_size, bias=False),
        ]
        self.aux = torch.nn.Sequential(*layers)
        # Residual Block (Bottom Part)
        
        self.BasicBlk1 = BasicBlock2Plus1D(in_channels, out_channels, stride=2, padding=1, downsample=True)
        self.BasicBlk2 = BasicBlock2Plus1D(out_channels, out_channels, stride=1, padding=1, downsample=False)
        self.relu      = torch.nn.ReLU(inplace=True)


    def forward(self, visual, tabular): 

        
        squeeze = self.global_avg_pool(visual)  # N, 512, 1, 1, 1 
        squeeze = torch.flatten(squeeze, 1)     # N, 512
        squeeze = torch.cat((squeeze, tabular), dim=1)

        affine = self.aux(squeeze)              # N, 2*split_size

        v_scale, v_shift = torch.split(affine, self.split_size, dim=1)  # N, split_size each
        v_scale = v_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # N, split_size, 1, 1, 1
        v_shift = v_shift.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # N, split_size, 1, 1, 1
        visual = visual * v_scale + v_shift


        x = self.BasicBlk1(visual)
        x = self.BasicBlk2(x)
        return x
