import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.optim import AdamW
from tqdm import tqdm


import detectors
import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import math
import copy





device="cuda"
resnet_model = timm.create_model("resnet18_cifar100", pretrained=True)
# print(resnet_model)



class TransformerConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride # Store stride
        self.num_heads = num_heads

        assert in_channels % num_heads == 0, "in_channels должно делиться на num_heads"
        self.head_dim = in_channels // num_heads

        # Проекции для MHA
        self.W_Q = nn.Parameter(torch.randn(num_heads, in_channels, self.head_dim))
        self.W_K = nn.Parameter(torch.randn(num_heads, in_channels, self.head_dim))
        self.W_V = nn.Parameter(torch.empty(num_heads, in_channels, self.head_dim))
        self.W_O = nn.Parameter(torch.randn(num_heads * self.head_dim, in_channels))

        # FFN
        hidden_dim = self.in_channels * 4 # Define hidden dimension, based on input channels
        self.ffn1 = nn.Linear(self.in_channels, hidden_dim)
        self.ffn2 = nn.Linear(hidden_dim, self.out_channels)

        # Нормализация и shortcut
        self.norm1 = nn.LayerNorm(in_channels,eps=1e-5)
        self.norm2 = nn.LayerNorm(out_channels,eps=1e-5)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

        # Main path spatial reduction if stride > 1
        self.main_path_spatial_reduction = nn.Sequential()
        if stride != 1:
             # This conv reduces spatial dimensions of the main path output
             self.main_path_spatial_reduction = nn.Sequential(
                 nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                 nn.BatchNorm2d(out_channels) # Add BN for consistency with shortcut
             )

        self._init_weights()
        self.scale_norm1 = nn.Parameter(torch.tensor(0.1))   
        self.scale_attn = nn.Parameter(torch.tensor(0.1))     
        self.scale_ffn = nn.Parameter(torch.tensor(0.1))      
        self.scale_norm2 = nn.Parameter(torch.tensor(0.01)) 

    def _init_weights(self):
        nn.init.uniform_(self.W_Q, -0.01, 0.01)
        nn.init.uniform_(self.W_K, -0.01, 0.01)
        nn.init.uniform_(self.W_V, -0.01, 0.01)
        nn.init.xavier_uniform_(self.W_O)
        nn.init.zeros_(self.ffn1.bias)
        nn.init.zeros_(self.ffn2.bias)
        # Initialize main_path_spatial_reduction if it exists
        if isinstance(self.main_path_spatial_reduction, nn.Sequential) and len(self.main_path_spatial_reduction) > 0:
             nn.init.kaiming_normal_(self.main_path_spatial_reduction[0].weight, mode='fan_out', nonlinearity='relu')
             nn.init.constant_(self.main_path_spatial_reduction[1].weight, 1)
             nn.init.constant_(self.main_path_spatial_reduction[1].bias, 0)

    def load_conv_weights(self, conv1, conv2):
        with torch.no_grad():
            conv1_weights = conv1.weight  # [C_out, C_in, 3, 3]
            head_locations = [(i, j) for i in range(3) for j in range(3)]
            selected_heads = head_locations[:self.num_heads]
            
            for head_idx, (i, j) in enumerate(selected_heads):
                patch = conv1_weights[:, :, i, j]  # [C_out, C_in]
                usable_dim = min(self.head_dim, patch.shape[1])
                self.W_V[head_idx, :, :usable_dim] = patch.T[:self.in_channels, :usable_dim]


            # FFN
            conv2_weights = conv2.weight  # [C_out, C_in, 3, 3]
            ffn1_weights = conv2_weights.mean(dim=[2, 3]) # [C_out, C_in]
            if ffn1_weights.shape[1] != self.ffn1.weight.shape[1] or ffn1_weights.shape[0] != self.ffn1.weight.shape[0]:
                ffn1_weights = F.adaptive_avg_pool2d(ffn1_weights.unsqueeze(0), self.ffn1.weight.shape[:2]).squeeze(0)
            self.ffn1.weight.data = ffn1_weights

    def forward(self, x):


        identity = x
        B, C, H, W = x.shape
        
        # Attention part
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)* self.scale_norm1 #[B, H, W, C]
        x_attn = torch.zeros_like(x_norm)
        
        # Reshape for attention
        x_reshaped = x_norm.permute(0, 2, 3, 1).reshape(B * H * W, C)  #[B*H*W, C]
        
        # Multi-head attention
        Q = torch.matmul(x_reshaped, self.W_Q.reshape(-1, self.head_dim * self.num_heads))  #[B*H*W, C] * [C , self.head_dim * self.num_heads] = [B*H*W, num_heads * head_dim]
        K = torch.matmul(x_reshaped, self.W_K.reshape(-1, self.head_dim * self.num_heads))
        V = torch.matmul(x_reshaped, self.W_V.reshape(-1, self.head_dim * self.num_heads))
        
        # Multi-head attention
        Q = torch.matmul(x_reshaped, self.W_Q.reshape(-1, self.head_dim * self.num_heads))  #[B*H*W, C] * [C , self.head_dim * self.num_heads] = [B*H*W, num_heads * head_dim]
        K = torch.matmul(x_reshaped, self.W_K.reshape(-1, self.head_dim * self.num_heads))
        V = torch.matmul(x_reshaped, self.W_V.reshape(-1, self.head_dim * self.num_heads))
        
        Q = Q.view(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # [B, num_heads, H, W, head_dim]
        K = K.view(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        V = V.view(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        
        # Attention scores
        attn_scores = torch.einsum('bnhwd,bmhwd->bnmhw', Q, K) / (self.head_dim ** 0.5)
        attn_scores = attn_scores - attn_scores.amax(dim=2, keepdim=True)  # Стабильный softmax
        attn_weights = F.softmax(attn_scores.clamp(-20, 20), dim=2) 
        x_attn = torch.einsum('bnmhw,bmhwd->bnhwd', attn_weights, V)
        x_attn = x_attn.permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)  #[B, H, W, num_heads * head_dim]
        x_attn = torch.matmul(x_attn, self.W_O).permute(0, 3, 1, 2)*self.scale_attn  #[B, H, W, out_channels]

        
        # Residual connection
        x = x_norm + x_attn
        x = F.relu(x)

        # FFN part
        # x is [B, out_channels, H, W] here after attention and relu.
        x = x.permute(0, 2, 3, 1) # Shape [B, H, W, out_channels]

        # Apply FFN to the last dimension (out_channels)
        x = self.ffn1(x)*self.scale_ffn # Shape [B, H, W, hidden_dim]

        x = F.relu(x)
        x = self.ffn2(x)*self.scale_ffn # Shape [B, H, W, out_channels]


        # Permute back to [B, out_channels, H, W]
        x = x.permute(0, 3, 1, 2)

        # Final normalization and residual
        x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) *self.scale_norm2 #[B, out_channels, H, W]

        # Apply spatial reduction to the main path output if stride > 1
        if self.stride != 1:
             x = self.main_path_spatial_reduction(x) # Now x has shape [B, out_channels, H', W']

        # Second residual connection
        x += self.shortcut(identity)
        x = F.relu(x)

        return x



# layer=TransformerConvBlock(in_channels=64,out_channels=128,stride=2)
# layer.load_conv_weights(resnet_model.layer2[0].conv1, resnet_model.layer2[0].conv2)
# print(layer)




class TransformerModel(nn.Module):
    def __init__(self, resnet):
        super(TransformerModel, self).__init__()
        
        # начальный conv, bn, relu, maxpool
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.act1,
            resnet.maxpool
        )

        self.layer1 = self.make_layer(resnet.layer1,stride0=1,stride1=1)
        self.layer2 = self.make_layer(resnet.layer2,stride0=2,stride1=1)
        self.layer3 = self.make_layer(resnet.layer3,stride0=2,stride1=1)
        self.layer4 = self.make_layer(resnet.layer4,stride0=2,stride1=1)

        # Pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 100)  # CIFAR-100

    def make_layer(self, resnet_layer,stride0,stride1):
        layers = []
        for idx, block in enumerate(resnet_layer):
            in_c = block.conv1.in_channels
            out_c = block.conv2.out_channels
            if idx==0:
                trans_block = TransformerConvBlock(in_c, out_c, num_heads=8,stride=stride0)
                trans_block.load_conv_weights(block.conv1, block.conv2)
            elif idx==1:
                trans_block = TransformerConvBlock(in_c, out_c, num_heads=8,stride=stride1)
                trans_block.load_conv_weights(block.conv1, block.conv2)
            layers.append(trans_block)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model_resnet=resnet_model.to(device)
model_transformer = TransformerModel(resnet_model).to(device)