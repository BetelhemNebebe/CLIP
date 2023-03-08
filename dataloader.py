import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

class ImageDataset(Dataset):
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform
        
    def __len__(self):
        return len(self.image_list)
        
    def __getitem__(self, idx):

        # transform the image if needed
        #if self.transform:
        #    transformed_image = self.transform(image)
        #return transformed_image
        image_array = self.image_list[idx]
        return torch.from_numpy(image_array)