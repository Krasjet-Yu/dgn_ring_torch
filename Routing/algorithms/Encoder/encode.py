import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
	def __init__(self, din, hidden_dim=32):
		super(Encoder, self).__init__()
		self.fc = nn.Linear(din, hidden_dim)

	def forward(self, x):
		embedding = F.relu(self.fc(x))
		return embedding