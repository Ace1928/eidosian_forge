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
def update_sat_metrics(self, loss, preds, labels, batchsize):
    self.metrics['sat_exs'] += batchsize
    self.metrics['sat_loss'] += loss.item()
    a = self.opt['target_class']
    b = not self.opt['target_class']
    assert a in [0, 1]
    assert b in [0, 1]
    self.metrics['sat_tp'] += ((preds == labels) * (labels == a)).sum().item()
    self.metrics['sat_fp'] += ((preds != labels) * (labels == b)).sum().item()
    self.metrics['sat_tn'] += ((preds == labels) * (labels == b)).sum().item()
    self.metrics['sat_fn'] += ((preds != labels) * (labels == a)).sum().item()