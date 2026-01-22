import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from .utils import load_embeddings, AverageMeter
from .rnn_reader import RnnDocReader
from parlai.utils.logging import logger
def set_embeddings(self):
    if not self.opt.get('embedding_file'):
        logger.warning('[ WARNING: No embeddings provided. Keeping random initialization. ]')
        return
    logger.info('[ Loading pre-trained embeddings ]')
    embeddings = load_embeddings(self.opt, self.word_dict)
    logger.info('[ Num embeddings = %d ]' % embeddings.size(0))
    new_size = embeddings.size()
    old_size = self.network.embedding.weight.size()
    if new_size[1] != old_size[1]:
        raise RuntimeError('Embedding dimensions do not match.')
    if new_size[0] != old_size[0]:
        logger.warning('[ WARNING: Number of embeddings changed (%d->%d) ]' % (old_size[0], new_size[0]))
    self.network.embedding.weight.data = embeddings
    if self.opt['tune_partial'] > 0:
        if self.opt['tune_partial'] + 2 < embeddings.size(0):
            fixed_embedding = embeddings[self.opt['tune_partial'] + 2:]
            self.network.fixed_embedding = fixed_embedding