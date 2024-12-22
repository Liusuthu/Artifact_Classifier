import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        use_batch_norm=False,
        use_residual=False,
    ):
        """
        Convolutional block with batch normalization and ReLU activation
        ----------------------
        :param in_channels: channel number of input image 输入图像通道数
        :param out_channels: channel number of output image 输出图像通道数
        :param kernel_size: size of convolutional kernel 
        :param stride: stride of convolutional operation
        :param padding: padding of convolutional operation 
        :param use_batch_norm: whether to use batch normalization in convolutional layers 是否使用batch normalization
        :param use_residual: whether to use residual connection 是否使用残差连接
        """
        super().__init__()

        if use_batch_norm:
            bn2d = nn.BatchNorm2d
        else:
            # use identity function to replace batch normalization
            bn2d = nn.Identity

        self.use_residual = use_residual

        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn = bn2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out_1 = self.conv(x)
        out_2 = self.bn(out_1)
        if self.use_residual:
            out =out_2 + x  # 残差连接操作
        else:
            out=out_2
        out = self.relu(out)  # 使用激活函数
        return out


class Classifier(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        use_batch_norm=False,
        use_stn=False,
        dropout_prob=0,
    ):
        """
        Convolutional Neural Networks
        ----------------------
        :param in_channels: channel number of input image 输入图像通道数
        :param num_classes: number of classes for the classification task 分类任务中的类别数量
        :param use_batch_norm: whether to use batch normalization in convolutional layers and linear layers
        :param use_stn: whether to use spatial transformer network 是否使用STN
        :param dropout_prob: dropout ratio of dropout layer which ranges from 0 to 1 Dropout概率
        """
        super().__init__()

        if use_batch_norm:
            bn1d = nn.BatchNorm1d
        else:
            # use identity function to replace batch normalization
            bn1d = nn.Identity


        # input image with size [batch_size, in_channels, img_h, img_w]
        # Network structure:
        #            kernel_size  stride  padding  out_channels  use_residual
        # ConvBlock       5          1        2          32         False
        # ConvBlock       5          2        2          64         False
        # maxpool         2          2        0
        # ConvBlock       3          1        1          64         True
        # ConvBlock       3          1        1          128        False
        # maxpool         2          2        0
        # ConvBlock       3          1        1          128        True
        # dropout(p), where p is input parameter of dropout ratio

        self.conv_net = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=5, stride=1, padding=2, use_batch_norm=use_batch_norm,use_residual=False),
            ConvBlock(32, 64, kernel_size=5, stride=2, padding=2, use_batch_norm=use_batch_norm,use_residual=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1, use_batch_norm=use_batch_norm,use_residual=True),
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1, use_batch_norm=use_batch_norm,use_residual=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1, use_batch_norm=use_batch_norm,use_residual=True),
            nn.Dropout(dropout_prob)
        )


        # Network structure:
        #            out_channels
        # linear          256
        # activation
        # batchnorm
        # dropout(p), where p is input parameter of dropout ratio
        # linear       num_classes
        self.fc_net = nn.Sequential(
            nn.Linear(in_features=128*40*40, out_features=256),
            nn.ReLU(),  # 激活函数
            bn1d(256),  # BatchNormalization层
            nn.Dropout(p=dropout_prob),  # Dropout层，使用输入参数dropout_prob来设置比率
            nn.Linear(256, num_classes)  # 输出层，输出维度为num_classes（类别数量）
        )

    def forward(self, x):
        """
        Define the forward function
        :param x: input features with size [batch_size, in_channels, img_h, img_w]
        :return: output features with size [batch_size, num_classes]
        """
        x = self.stn(x)
        x_1=self.conv_net(x)
        x_1=x_1.view(x_1.size(0),-1)
        out=self.fc_net(x_1)

        return out

