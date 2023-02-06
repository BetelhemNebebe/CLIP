import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Compute_metrics():
    def __init__(self, feature1, feature2):
        super().__init__()
        self.feature1 = feature1
        self.feature2 = feature2

    def cosine_similarity(self):
        tensor1 = self.feature1
        tensor2 = torch.tensor(self.feature2)
        
        tensor1 = F.normalize(tensor1, dim=1)
        tensor2 = F.normalize(tensor2, dim=1)

        cos = nn.CosineSimilarity()
        output = cos(tensor1, tensor2)

        return output