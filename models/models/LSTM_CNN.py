import os
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
ConvMethod = "in_channel_is_embedding_dim"

# using output of LSTm as input for CNN

class LSTM_CNN(nn.Module):
	def __init__(self, **kwargs):
		super(LSTM_CNN, self).__init__()
		
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

		self.HIDDEN_DIM = kwargs["HIDDEN_DIM"]
		self.NUM_LAYERS = kwargs["NUM_LAYERS"]
		self.bidirectional = kwargs["BIDIRECTIONAL"]

		self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
		if self.MODEL == "static" or self.MODEL == "non-static" or self.MODEL == "multichannel":
			self.WV_MATRIX = kwargs["WV_MATRIX"]
			self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
		if self.MODEL == "static":
			self.embedding.weight.requires_grad = False
		elif self.MODEL == "multichannel":
			self.embedding2 == nn.Embedding(self.VOCAB_SIZE + 2, self_WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
			self.embedding2.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
			self.embedding2.weight.requires_grad = False
			self.IN_CHANNEL = 2

		conv_blocks = []
		for filter_size in self.FILTERS:
			maxpool_kernel_size = self.MAX_SENT_LEN - filter_size + 1
			if ConvMethod == "in_channel_is_embedding_dim":
				conv1d = nn.Conv1d(in_channels = self.HIDDEN_DIM, out_channels = self.FILTER_NUM, kernel_size = filter_size, stride = 1)
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

		self.LSTM = nn.LSTM(self.WORD_DIM, self.HIDDEN_DIM, dropout=self.DROPOUT_PROB, num_layers=self.NUM_LAYERS)
		self.hidden = self.init_hidden()

	def init_hidden(self):
		if self.USE_CUDA == True:
			return (Variable(torch.zeros(1 * self.NUM_LAYERS, self.BATCH_SIZE, self.HIDDEN_DIM)).cuda(),
				Variable(torch.zeros(1 * self.NUM_LAYERS, self.BATCH_SIZE, self.HIDDEN_DIM)).cuda())
		else:
			return (Variable(torch.zeros(1 * self.NUM_LAYERS, self.BATCH_SIZE, self.HIDDEN_DIM)),
				Variable(torch.zeros(1 * self.NUM_LAYERS, self.BATCH_SIZE, self.HIDDEN_DIM)))

	def forward(self, x):
		x = self.embedding(x)
		x = x.transpose(0,1)
		x, self.hidden = self.LSTM(x, self.hidden)
		x = x.transpose(0,1)
		if ConvMethod == "in_channel_is_embedding_dim":
			x = x.transpose(1,2)	      # (batch, embed_dim, sent_len)
		else:
			x = x.view(x.size(0), 1, -1)  # (batch, 1, sent_len*embed_dim)
		x_list = [conv_block(x) for conv_block in self.conv_blocks]
		out = torch.cat(x_list, 2)
		out = out.view(out.size(0), -1)
		out = F.dropout(out, p=self.DROPOUT_PROB, training=self.training)
		out = self.fc(out)
		return out
		
