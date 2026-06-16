from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np

class ProcessData:
    def __init__(self, data):
        self.data = data

    def process(self):
        
        if torch.is_tensor(self.data):
            self.data = self.data.numpy()

        coords = self.data.reshape(21, 3)

        wrist_coord = coords[0]

        relative_coords = coords - wrist_coord

        distances = np.sqrt(np.sum(relative_coords ** 2, axis=1))
        max_distance = np.max(distances)

        if max_distance > 0:
            relative_coords = relative_coords / max_distance

        return relative_coords.flatten()