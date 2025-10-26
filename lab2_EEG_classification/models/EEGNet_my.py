#%%
import torch.nn as nn

# TODO implement EEGNet model
class EEGNet(nn.Module):
    def __init__(self, activation='elu'):
        super(EEGNet, self).__init__()
        # 根據 activation 參數選擇激活函數
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'elu':
            self.activation = nn.ELU(alpha=1.0)
        else:
            raise ValueError("Unsupported activation function")
        
        # FirstConv: Temporal Convolution
        # Input: (batch, 1, 2, 750) -> Output: (batch, 16, 2, 750)
        self.firstconv = nn.Sequential(          # 按順序執行多個層
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), 
                      padding=(0, 25), bias=False),
            nn.BatchNorm2d(16)
        )
        
        # DepthwiseConv: Spatial Convolution
        # Input: (batch, 16, 2, 750) -> Output: (batch, 32, 1, 750)
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), 
                      groups=16, bias=False),
            nn.BatchNorm2d(32),
            self.activation,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=0.25)
        )
        
        # SeparableConv: Depthwise Separable Convolution
        # Input: (batch, 32, 1, 187) -> Output: (batch, 32, 1, 187)
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), 
                      padding=(0, 7), groups=32, bias=False),
            nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            self.activation,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=0.25)
        )
        
        # Classifier
        # Input: (batch, 32, 1, 23) -> Output: (batch, 2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 1 * 23, 2, bias=True)
        )
    
    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.classifier(x)
        return x

#%%
# (Optional) implement DeepConvNet model
class DeepConvNet(nn.Module):
    def __init__(self):
        super(DeepConvNet, self).__init__()
        pass

    def forward(self, x):
        pass