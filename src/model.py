import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiOmicsNet(nn.Module):
    def __init__(self, input_dims, hidden_dim=128, num_classes=5):
        super(MultiOmicsNet, self).__init__()
        self.omics_branches = nn.ModuleDict({
            omic: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for omic, dim in input_dims.items()
        })
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * len(input_dims), 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_dict):
        features = [branch(x_dict[omic]) for omic, branch in self.omics_branches.items()]
        concat = torch.cat(features, dim=1)
        return self.classifier(concat)
