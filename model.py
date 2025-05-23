import torch
import torch.nn as nn

class IntrusionDetector(nn.Module):
    def __init__(self, dropout =0.2):
        super(IntrusionDetector, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=13, out_features=64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p = dropout),

            nn.Linear(in_features=64, out_features=32), 
            # nn.BatchNorm1d(32),
            nn.ReLU(),        
            
            nn.Linear(in_features=32, out_features=1),
            # nn.Sigmoid()
        )
    def forward(self, x):
        x = self.network(x)
        return x