import os
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
ConvMethod = "in_channel_is_embedding_dim"
class CNN(nn.Module):
	def __init__(self, **kwargs):
		super(CNN, self).__init__()
		
		self.MODEL = kwargs["MODEL"]
		self.BATCH_SIZE = kwargs["BATCH_SIZE"]
		self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
		self.WORD_DIM = kwargs["WORD_DIM"]
		self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
		self.CLASS_SIZE = kwargs["CLASS_SIZE"]
		self.FILTERS = kwargs["FILTERS"]
		self.FILTER_NUM = kwargs["FILTER_NUM"]
		self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
		self.IN_CHANNEL = 1
		self.USE_CUDA = True
		print (self.VOCAB_SIZE)	
		#assert (len(self.FILTERS) == len(self.FILTER_NUM))
		self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
		if self.MODEL == "static" or self.MODEL == "non-static":
			self.WV_MATRIX = kwargs["WV_MATRIX"]
			self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
		if self.MODEL == "static":
			self.embedding.weight.requires_grad = False

		conv_blocks = []
		conv_blocks1 = []
		for filter_size in self.FILTERS:
			maxpool_kernel_size = self.MAX_SENT_LEN - filter_size + 1
			if ConvMethod == "in_channel_is_embedding_dim":
				conv1d = nn.Conv1d(in_channels = self.WORD_DIM, out_channels = self.FILTER_NUM, kernel_size = filter_size, stride = 1)
				conv2 = nn.Conv1d(in_channels = self.FILTER_NUM , out_channels = self.FILTER_NUM, kernel_size = filter_size/2, stride = 1)
			else:
				conv1d = nn.Conv1d(in_channels = 1, out_channels = self.FILTER_NUM, kernel_size = filter_size * self.WORD_DIM, stride = self.WORD_DIM)
				conv2 = nn.Conv1d(in_channels = self.FILTER_NUM, out_channels = self.FILTER_NUM, kernel_size = filter_size * self.WORD_DIM, stride = self.WORD_DIM)

			component = nn.Sequential(
				conv1d,
				nn.ReLU(),
				nn.MaxPool1d(kernel_size = maxpool_kernel_size),
			)
			if self.USE_CUDA:
				component = component.cuda()
					
			conv_blocks.append(component)
		self.conv_blocks = nn.ModuleList(conv_blocks)
		self.fc = nn.Linear(self.FILTER_NUM * len(self.FILTERS), self.CLASS_SIZE)
	
	def forward(self, x):
		x = self.embedding(x)
		if ConvMethod == "in_channel_is_embedding_dim":
			x = x.transpose(1,2)	      # (batch, embed_dim, sent_len)
		else:
			x = x.view(x.size(0), 1, -1)  # (batch, 1, sent_len*embed_dim)
		x_list = [conv_block(x) for conv_block in self.conv_blocks]
		out = torch.cat(x_list, 2)
		#out = out.transpose(1,2)	      # (batch, embed_dim, sent_len)
		out = out.view(out.size(0), -1)
		#feature_extracted = out
		out = F.dropout(out, p=self.DROPOUT_PROB, training=self.training)
		out = self.fc(out)
		return out 

