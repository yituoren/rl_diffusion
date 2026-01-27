from torchvision.datasets import CIFAR10, ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

def get_cifar10_data_loaders(root: str, batch_size: int = 128, num_workers: int = 4):
    train_dataset = CIFAR10(root=root, train=True, download=True, transform=transforms.ToTensor())
    test_dataset = CIFAR10(root=root, train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def get_cifar10_128_data_loaders(root: str, batch_size: int = 128, num_workers: int = 4):
    train_dataset = ImageFolder(root=f"{root}/train", transform=transforms.ToTensor())
    test_dataset = ImageFolder(root=f"{root}/test", transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader