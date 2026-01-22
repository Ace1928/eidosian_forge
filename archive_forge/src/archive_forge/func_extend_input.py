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
def extend_input(self, batch):
    """
        Extend the input.
        """
    pad_tensor = torch.zeros(1, batch.text_vec.size(1)).long().cuda()
    text_vec = torch.cat([batch.text_vec, pad_tensor], 0)
    batch = batch._replace(text_vec=text_vec)
    if batch.label_vec is not None:
        pad_tensor = torch.zeros(1, batch.label_vec.size(1)).long().cuda()
        label_vec = torch.cat([batch.label_vec, pad_tensor], 0)
        batch = batch._replace(label_vec=label_vec)
    if batch.candidates is not None:
        dummy_list = [['None'] for _ in range(len(batch.candidates[0]))]
        batch = batch._replace(candidates=batch.candidates + [dummy_list])
        new_vecs = batch.candidate_vecs + [[torch.zeros(1).long() for _ in range(len(batch.candidate_vecs[0]))]]
        batch = batch._replace(candidate_vecs=new_vecs)
    return batch