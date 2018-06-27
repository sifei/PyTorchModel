import os
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
ConvMethod = "in_channel_is_embedding_dim"

class GRU(nn.Module):
	def __init__(self, **kwargs):
		super(GRU, self).__init__()
		
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
		self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM)
		if self.MODEL == "static" or self.MODEL == "non-static":
			self.WV_MATRIX = kwargs["WV_MATRIX"]
			self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
		if self.MODEL == "static":
			self.embedding.weight.requires_grad = False
		self.GRU = nn.GRU(self.WORD_DIM, self.HIDDEN_DIM, dropout=self.DROPOUT_PROB, num_layers=self.NUM_LAYERS)
		self.hidden2label = nn.Linear(self.HIDDEN_DIM, self.CLASS_SIZE)
		self.hidden = self.init_hidden()

	def init_hidden(self):
		if self.bidirectional == True:
			return Variable(torch.zeros(2 * self.NUM_LAYERS, self.BATCH_SIZE, self.HIDDEN_DIM)).cuda()
		else:
			return Variable(torch.zeros(1 * self.NUM_LAYERS, self.BATCH_SIZE, self.HIDDEN_DIM)).cuda()


	def forward(self, x):
		x = self.embedding(x)
		x = x.transpose(0,1)
		lstm_out, self.hidden = self.GRU(x,self.hidden)
		y = self.hidden2label(lstm_out[-1])
		return y#F.log_softmax(y)	
