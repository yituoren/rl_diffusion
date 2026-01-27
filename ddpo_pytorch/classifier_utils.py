import torch.nn as nn
import torch
import os
from torchvision import models
from .reward_models.classifier import get_classifier

class ClassifierEnsemble(nn.Module):
    def __init__(self, classifiers):
        super(ClassifierEnsemble, self).__init__()
        self.classifiers = nn.ModuleList(classifiers)

    def forward(self, x):
        # Average the logits from all classifiers
        all_logits = [clf(x) for clf in self.classifiers]
        avg_logits = torch.stack(all_logits).mean(dim=0)
        return avg_logits

def load_classifier_ensemble(config, device):
    """Load one or more classifiers into an ensemble"""
    # Ensure we are working with a list
    checkpoint_paths = config.classifier_checkpoint
    if isinstance(checkpoint_paths, str):
        checkpoint_paths = [checkpoint_paths]
    
    loaded_classifiers = []
    
    for path in checkpoint_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Classifier checkpoint not found: {path}")
        
        # You might need to adjust 'config.classifier' if your ensemble 
        # uses different architectures (e.g., ResNet18 and MobileNet)
        classifier = get_classifier(config.classifier, device=device)
        
        checkpoint = torch.load(path, map_location=device)
        # Handle different checkpoint formats if necessary
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        classifier.load_state_dict(state_dict)
        
        classifier.eval()
        for param in classifier.parameters():
            param.requires_grad = False
        
        loaded_classifiers.append(classifier)
        print(f"Successfully loaded classifier from {path}")
    
    return ClassifierEnsemble(loaded_classifiers).to(device)