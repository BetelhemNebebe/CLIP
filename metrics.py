import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

class Compute_metrics():
    def __init__(self, feature1, feature2):
        super().__init__()
        self.feature1 = feature1
        self.feature2 = feature2

    def similarity(self):
        tensor1 = self.feature1
        tensor2 = torch.tensor(self.feature2)
        
        tensor1 = F.normalize(tensor1, dim=1)
        tensor2 = F.normalize(tensor2, dim=1)

        cos = nn.CosineSimilarity()
        output = cos(tensor1, tensor2)

        return output
    
    def logits(self):
        # normalized features
        image_features = self.feature1 / self.feature1.norm(dim=1, keepdim=True)
        audio_features = self.feature2 / self.feature2.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        logit_scale = logit_scale.exp()
        logits_per_image = logit_scale * image_features @ audio_features.t()
        logits_per_audio = logits_per_image.t()
        probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()

        #similarity = (100.0 * image_features @ audio_features.T).softmax(dim=-1)

        return probs #logits_per_image, logits_per_audio

    def pair_cosine_similarity(self):
        similarity = cosine_similarity(self.feature1, self.feature2)
        return similarity