import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from parlai.core.torch_agent import TorchAgent, Output
from parlai.utils.misc import round_sigfigs, warn_once
from parlai.utils.torch import padded_tensor
from parlai.agents.transformer.transformer import TransformerRankerAgent
from .feedback_classifier.feedback_classifier import FeedbackClassifierRegex
from .modules import SelfFeedingModel
def update_dia_metrics(self, loss, ranks, label_inds, batchsize):
    self.metrics['dia_exs'] += batchsize
    self.metrics['dia_loss'] += loss.item()
    if label_inds is not None:
        for b in range(batchsize):
            rank = (ranks[b] == label_inds[b]).nonzero().item()
            self.metrics['dia_rank'] += 1 + rank
            self.metrics['dia_correct'] += rank == 0