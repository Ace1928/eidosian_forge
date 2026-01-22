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
def batchify(self, *args, **kwargs):
    """
        Override batchify options for seq2seq.
        """
    kwargs['sort'] = True
    batch = super().batchify(*args, **kwargs)
    obs_batch = args[0]
    sort = kwargs['sort']
    is_valid = lambda obs: 'text_vec' in obs or 'image' in obs
    if len(obs_batch) == 0:
        return Batch()
    valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if is_valid(ex)]
    if len(valid_obs) == 0:
        return Batch()
    valid_inds, exs = zip(*valid_obs)
    xs, x_lens = (None, None)
    if any(('text_vec' in ex for ex in exs)):
        _xs = [ex.get('text_vec', self.EMPTY) for ex in exs]
        xs, x_lens = padded_tensor(_xs, self.NULL_IDX, self.use_cuda)
        if sort:
            sort = False
            xs, x_lens, valid_inds, exs = argsort(x_lens, xs, x_lens, valid_inds, exs, descending=True)
    history = [ConvAI2History(ex['full_text'], dictionary=self.dict) for ex in exs]
    ctrl_vec = get_ctrl_vec(exs, history, self.control_settings)
    if self.use_cuda and ctrl_vec is not None:
        ctrl_vec = ctrl_vec.cuda()
    ControlBatch = namedtuple('Batch', tuple(batch.keys()) + ('ctrl_vec', 'history'))
    batch = ControlBatch(ctrl_vec=ctrl_vec, history=history, **dict(batch))
    return batch