import torch.nn as nn
import torch.nn.functional as F
import torch._utils
from torch.nn.modules.utils import _pair, _quadruple

def initialize_weights(module, slope=0, init_mode='fan_out', activation='relu', init_bias=0, dist='normal'):
    """
    Initializes the weights of a module according to the He (Kaiming) initialization method.

    Parameters:
    - module (nn.Module): The module to be initialized.
    - slope (float): The negative slope of the rectifier used after this layer. Default is 0.
    - init_mode (str): Can be 'fan_in' (preserves the magnitude of the variance in the forward pass)
                       or 'fan_out' (preserves the magnitudes in the backward pass). Default is 'fan_out'.
    - activation (str): Specifies the non-linearity used after this layer. Default is 'relu'.
    - init_bias (float): The value to initialize the bias with. Default is 0.
    - dist (str): Specifies the distribution to use for weight initialization ('uniform' or 'normal'). Default is 'normal'.
    """

    assert dist in ['uniform', 'normal'], "Distribution must be either 'uniform' or 'normal'."

    if dist == 'uniform':
        init.kaiming_uniform_(module.weight, a=slope, mode=init_mode, nonlinearity=activation)
    elif dist == 'normal':
        init.kaiming_normal_(module.weight, a=slope, mode=init_mode, nonlinearity=activation)

    if hasattr(module, 'bias') and module.bias is not None:
        init.constant_(module.bias, init_bias)

        
        
class HSA(nn.Module):
    def __init__(self, planes, kernel_size=(5, 5), padding=2, ksize=3, do_padding=False, bias=False):
        super(HSA, self).__init__()
        self.global_branch = GlobalAttentionBranch(inplanes=planes[0], planes=planes[-1])  # Global Branch
        self.local_branch = nn.Sequential(  # Local Branch (Combining Self_integration and SelfCorrelationComputation)
            SelfCorrelationComputation(kernel_size=kernel_size, padding=padding),
            FeatureSelfIntegration(planes=planes, ksize=ksize, do_padding=do_padding, bias=bias)
        )

    def forward(self, x):
        # Process input through the global branch
        x_global = self.global_branch(x)
        # Process input through the local branch
        x_local = self.local_branch(x)
        # Fusion of global and local features
        x_self_att = x_global + x_local  # This fusion method can be adjusted as needed
        return x_self_att




class GlobalAttentionBranch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super(GlobalAttentionBranch, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = out_channels // 2
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        ratio = 4

        # Query and Value Convolutions for Right Branch
        self.query_conv_right = nn.Conv2d(in_channels, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.value_conv_right = nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, stride=stride, padding=0,
                                          bias=False)

        # Upscaling convolution for Right Branch
        self.feature_upscale_right = nn.Sequential(
            nn.Conv2d(self.mid_channels, self.mid_channels // ratio, kernel_size=1),
            nn.LayerNorm([self.mid_channels // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_channels // ratio, out_channels, kernel_size=1)
        )

        # Query and Value Convolutions for Left Branch
        self.query_conv_left = nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, stride=stride, padding=0,
                                         bias=False)
        self.value_conv_left = nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, stride=stride, padding=0,
                                         bias=False)

        # Pooling and Activation
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        query = self.query_conv_right(x).view(batch, 1, height * width)
        query = self.softmax(query)

        value = self.value_conv_right(x).view(batch, self.mid_channels, height * width)
        context = torch.matmul(value, query.transpose(1, 2)).unsqueeze(-1)
        context = self.feature_upscale_right(context).view(batch, self.out_channels, height, width)

        return self.sigmoid(context) * x

    def channel_pool(self, x):
        query = self.query_conv_left(x)
        value = self.value_conv_left(x).view(x.size(0), self.mid_channels, -1)
        query = self.avg_pool(query).view(x.size(0), self.mid_channels, 1).permute(0, 2, 1)

        context = torch.matmul(query, value).view(x.size(0), 1, x.size(2), x.size(3))
        return self.sigmoid(context) * x

    def forward(self, x):
        x = self.spatial_pool(x)
        x = self.channel_pool(x)
        return x

class FeatureSelfIntegration(nn.Module):
    """
    Enhances feature representation through self-integration using convolutional layers.
    """
    def __init__(self, channels=[640, 64, 64, 64, 640], stride=1, kernel_size=3, do_padding=False, bias=False):
        super(FeatureSelfIntegration, self).__init__()
        self.kernel_size = _quadruple(kernel_size) if isinstance(kernel_size, int) else kernel_size
        padding = (0, self.kernel_size[2] // 2, self.kernel_size[3] // 2) if do_padding else (0, 0, 0)

        # Initial convolution to reduce dimension
        self.conv_in = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=1, bias=bias, padding=0),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True)
        )

        # 3D convolutions for self-integration
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(channels[1], channels[2], (1, self.kernel_size[2], self.kernel_size[3]),
                      stride=stride, bias=bias, padding=padding),
            nn.BatchNorm3d(channels[2]),
            nn.ReLU(inplace=True)
        )

        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(channels[2], channels[3], (1, self.kernel_size[2], self.kernel_size[3]),
                      stride=stride, bias=bias, padding=padding),
            nn.BatchNorm3d(channels[3]),
            nn.ReLU(inplace=True)
        )

        # Final convolution to upscale dimension
        self.conv_out = nn.Sequential(
            nn.Conv2d(channels[3], channels[4], kernel_size=1, bias=bias, padding=0),
            nn.BatchNorm2d(channels[4])
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.conv_in(x)  # Initial dimension reduction

        # Adjusting view for 3D convolution
        x = x.view(b, c, h*w, 1, 1)  # Adjust the shape appropriately based on the actual operation

        x = self.conv3d_1(x)  # Self-integration step 1
        x = self.conv3d_2(x)  # Self-integration step 2

        # Reshaping back to 2D convolution input
        x = x.view(b, -1, h, w)
        x = self.conv_out(x)  # Dimension upscale

        return x


class SelfCorrelationComputation(nn.Module):
    """
        Computes self-correlation of features within an image to enhance local feature representations.
        """

    def __init__(self, kernel_size=(5, 5), padding=2):
        super(SelfCorrelationComputation, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, padding=self.padding)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # Initial activation and normalization
        x = self.activation(x)
        x_normalized = F.normalize(x, dim=1, p=2)

        # Unfolding and computing self-correlation
        unfolded_x = self.unfold(x_normalized)  # Unfolding the input
        b, c, h, w = x.size()
        k_h, k_w = self.kernel_size

        # Reshaping unfolded input for self-correlation computation
        unfolded_x = unfolded_x.view(b, c, k_h, k_w, -1)
        unfolded_x = unfolded_x.permute(0, 1, 4, 2, 3)  # Rearranging dimensions to (b, c, h*w, k_h, k_w)

        # Performing element-wise multiplication between the unfolded input and the identity matrix
        identity = x_normalized.view(b, c, 1, h, w)
        self_correlation = unfolded_x * identity

        # Rearranging the self-correlation output to match input dimensions
        self_correlation = self_correlation.view(b, c, h, w, k_h, k_w)
        self_correlation = self_correlation.permute(0, 1, 3, 4, 2, 5).contiguous()  # Back to (b, c, h, w, u, v)

        return self_correlation
