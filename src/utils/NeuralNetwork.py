import torch
import torch.nn as nn

from utils.extract_class import label_dict_from_config_file

class NeuralNetwork(nn.Module):
  def __init__(self):
    super(NeuralNetwork, self).__init__()
    self.flatten = nn.Flatten()
    list_labels = label_dict_from_config_file("../Dataset/hand_gesture.yaml")

    self.linear_relu_stack = nn.Sequential(

        # Layer 1
        nn.Linear(63, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.15),

        # Layer 2
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.15),

        # Output layer
        nn.Linear(32, len(list_labels))
    )

  def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits

  def predict(self, x, threshold = 0.8):
    logits = self(x)
    softmax_prob = nn.Softmax(dim=1)(logits)
    chosen_id = torch.argmax(softmax_prob, dim = 1)
    return torch.where(softmax_prob[0, chosen_id] > threshold, chosen_id, -1)

  def predict_with_known_class(self, x):
    logits = self(x)
    softmax_prob = nn.Softmax(dim=1)(logits)
    return torch.argmax(softmax_prob, dim = 1)

  def score(self, logits):
    return -torch.amax(logits, dim = 1)