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
def encode_candidates(self, cands):
    """
        Encodes a tensor of vectorized candidates.

        :param cands: a [bs, seq_len] or [bs, num_cands, seq_len](?) of vectorized
            candidates
        """
    return self.model.encode_dia_y(cands)