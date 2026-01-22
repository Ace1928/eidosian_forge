from parlai.core.torch_agent import TorchAgent, Output, Batch
from parlai.utils.misc import round_sigfigs
from parlai.utils.torch import padded_tensor, argsort, neginf
from .modules import Seq2seq, opt_to_kwargs
from .util import ConvAI2History, show_beam_cands, reorder_extrep2gram_qn
from .controls import (
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, Counter
from operator import attrgetter
import os
import math
import json
import tempfile
import copy
def get_top_hyp(self):
    """
        Get single best hypothesis.

        :return: hypothesis sequence and the final score
        """
    top_hypothesis_tail = self.get_rescored_finished(n_best=1)[0]
    return (self.get_hyp_from_finished(top_hypothesis_tail), top_hypothesis_tail.score)