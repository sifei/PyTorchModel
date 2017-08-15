import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class BoW(nn.Module):
    def __init__(self, emb, nc=2, dim=300, p_drop=0.5, hidden=2048,
                penalty=0., train_emb=True, decay=0., clip=None, lr=0.001):
        """
            Args:
                vocab_size: Number of words in vocab
                emb: Word embeddings matrix (num_words X word_dimension)
                nc: Number of classes
                dim: Dimensionality of word embeddings
                p_drop: Dropout probability
                hidden: Hidden layer dimensions
                penalty: l2 regularization param
                train_emb: Boolean if the embeedings should be trained
                decay: Learning rate decay parameter
                clip: Gradient Clipping parameter (None == don't clip)
                lr: Learning rate
        """
        super(BoW, self).__init__()
        self.linear = nn.Linear(emb.shape[0], nc)

    def forward(self, emb):
        embeds = torch.from_numpy(emb)
        log_probs = F.log_softmax(self.linear(embeds))
        return log_probs
