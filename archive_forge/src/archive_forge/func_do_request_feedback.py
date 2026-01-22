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
def do_request_feedback(self, positivity):
    """
        Decide whether to request an feedback this turn.
        """
    if not self.opt['request_feedback'] or len(self.history.history_strings) == 1:
        return False
    else:
        return positivity < self.opt['rating_threshold']