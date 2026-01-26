"""
Flexible CNN Classifier

Supports both MNIST (grayscale) and CIFAR-10 (RGB) with configurable architecture.
Also supports ResNet architectures with optional pretrained weights.
"""

import torch
import torch.nn as nn
from torchvision import models


class FlexibleClassifier(nn.Module):
    """
    Flexible CNN classifier that adapts to input channels and image sizes.
    
    Args:
        in_channels: Number of input channels (1 for MNIST, 3 for CIFAR-10)
        num_classes: Number of output classes (default: 10)
        hidden_dims: List of hidden layer dimensions (default: [64, 128, 256])
        dropout: Dropout probability (default: 0.3)
    """
    
    def __init__(self, in_channels=1, num_classes=10, hidden_dims=None, dropout=0.3):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        # Build convolutional layers
        conv_layers = []
        prev_channels = in_channels
        
        for i, dim in enumerate(hidden_dims):
            conv_layers.extend([
                nn.Conv2d(prev_channels, dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ])
            prev_channels = dim
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate the output size after convolutions
        # For MNIST (28x28): after 3 max pools -> 3x3
        # For CIFAR-10 (32x32): after 3 max pools -> 4x4
        # We'll use adaptive pooling to handle this flexibly
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dims[-1], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, H, W)
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = self.fc(x)
        return x


class ResNetClassifier(nn.Module):
    """
    ResNet-based classifier with optional pretrained weights.
    
    Args:
        arch: ResNet architecture ('resnet18', 'resnet34', 'resnet50')
        in_channels: Number of input channels (1 for MNIST, 3 for CIFAR-10)
        num_classes: Number of output classes (default: 10)
        pretrained: Whether to use pretrained ImageNet weights (default: False)
        freeze_backbone: Whether to freeze backbone layers (default: False)
    """
    
    def __init__(self, arch='resnet18', in_channels=3, num_classes=10, 
                 pretrained=False, freeze_backbone=False, dropout=0.0):
        super().__init__()
        
        self.arch = arch
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        
        # Load ResNet model
        if arch == 'resnet18':
            if pretrained:
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet18(weights=None)
            feature_dim = 512
        elif arch == 'resnet34':
            if pretrained:
                self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet34(weights=None)
            feature_dim = 512
        elif arch == 'resnet50':
            if pretrained:
                self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet50(weights=None)
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown ResNet architecture: {arch}")
        
        # Modify first conv layer if input channels != 3
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Replace the final fully connected layer
        self.backbone.fc = nn.Linear(feature_dim, num_classes)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if 'fc' not in name:  # Don't freeze the final classifier layer
                    param.requires_grad = False
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, H, W)
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True


class MNISTClassifier(nn.Module):
    """
    Original MNIST classifier for backward compatibility.
    Simple CNN architecture optimized for 28x28 grayscale images.
    """
    
    def __init__(self):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14x14
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 7x7
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 3x3
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x


def get_classifier(config, device='cuda'):
    if config.model_type == 'flexible':
        return FlexibleClassifier(
            in_channels=config.in_channels,
            num_classes=config.num_classes,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout
        ).to(device)
    elif config.model_type == 'resnet':
        return ResNetClassifier(
            arch=config.resnet_arch,
            in_channels=config.in_channels,
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            freeze_backbone=config.freeze_backbone,
        ).to(device)
    else:
        raise ValueError(f"Unknown classifier model type: {config.model_type}")


if __name__ == '__main__':
    # Test MNIST classifier
    print("Testing MNIST classifier:")
    mnist_model = FlexibleClassifier(in_channels=1, num_classes=10)
    mnist_input = torch.randn(4, 1, 28, 28)
    mnist_output = mnist_model(mnist_input)
    print(f"  Input shape: {mnist_input.shape}")
    print(f"  Output shape: {mnist_output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in mnist_model.parameters()):,}")
    
    # Test CIFAR-10 classifier
    print("\nTesting CIFAR-10 classifier:")
    cifar_model = FlexibleClassifier(in_channels=3, num_classes=10)
    cifar_input = torch.randn(4, 3, 32, 32)
    cifar_output = cifar_model(cifar_input)
    print(f"  Input shape: {cifar_input.shape}")
    print(f"  Output shape: {cifar_output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in cifar_model.parameters()):,}")
    
    # Test ResNet18 (no pretrained)
    print("\nTesting ResNet18 (no pretrained):")
    resnet18_model = ResNetClassifier(arch='resnet18', in_channels=3, num_classes=10, pretrained=False)
    resnet18_output = resnet18_model(cifar_input)
    print(f"  Input shape: {cifar_input.shape}")
    print(f"  Output shape: {resnet18_output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in resnet18_model.parameters()):,}")
    
    # Test ResNet18 (pretrained)
    print("\nTesting ResNet18 (pretrained):")
    resnet18_pretrained = ResNetClassifier(arch='resnet18', in_channels=3, num_classes=10, pretrained=True)
    resnet18_pretrained_output = resnet18_pretrained(cifar_input)
    print(f"  Input shape: {cifar_input.shape}")
    print(f"  Output shape: {resnet18_pretrained_output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in resnet18_pretrained.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in resnet18_pretrained.parameters() if p.requires_grad):,}")
    
    # Test ResNet50
    print("\nTesting ResNet50:")
    resnet50_model = ResNetClassifier(arch='resnet50', in_channels=3, num_classes=10, pretrained=False)
    resnet50_output = resnet50_model(cifar_input)
    print(f"  Input shape: {cifar_input.shape}")
    print(f"  Output shape: {resnet50_output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in resnet50_model.parameters()):,}")
    
    # Test ResNet with MNIST (1 channel)
    print("\nTesting ResNet18 with MNIST (1 channel):")
    resnet18_mnist = ResNetClassifier(arch='resnet18', in_channels=1, num_classes=10, pretrained=False)
    resnet18_mnist_output = resnet18_mnist(mnist_input)
    print(f"  Input shape: {mnist_input.shape}")
    print(f"  Output shape: {resnet18_mnist_output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in resnet18_mnist.parameters()):,}")
    
    # Test original MNIST classifier
    print("\nTesting original MNIST classifier:")
    old_mnist_model = MNISTClassifier()
    old_mnist_output = old_mnist_model(mnist_input)
    print(f"  Input shape: {mnist_input.shape}")
    print(f"  Output shape: {old_mnist_output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in old_mnist_model.parameters()):,}")
    
    # # Test factory function
    # print("\nTesting factory function:")
    # factory_model = get_classifier('resnet', arch='resnet34', in_channels=3, num_classes=10, pretrained=True)
    # factory_output = factory_model(cifar_input)
    # print(f"  Model type: ResNet34 (pretrained)")
    # print(f"  Output shape: {factory_output.shape}")
    # print(f"  Parameters: {sum(p.numel() for p in factory_model.parameters()):,}")