import torch
import torch.nn as nn
from parlai.agents.transformer.modules import TransformerEncoder
def init_embeddings(self):
    embeddings = nn.Embedding(self.vocab_size, self.opt['embedding_size'], padding_idx=self.pad_idx)
    nn.init.normal_(embeddings.weight, mean=0, std=self.opt['embedding_size'] ** (-0.5))
    nn.init.constant_(embeddings.weight[self.pad_idx], 0)
    return embeddings