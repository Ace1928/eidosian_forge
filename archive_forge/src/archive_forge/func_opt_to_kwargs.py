import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from parlai.utils.torch import NEAR_INF
def opt_to_kwargs(opt):
    """
    Get kwargs for seq2seq from opt.
    """
    kwargs = {}
    for k in ['numlayers', 'dropout', 'bidirectional', 'rnn_class', 'lookuptable', 'decoder', 'numsoftmax', 'attention', 'attention_length', 'attention_time', 'input_dropout', 'control_settings']:
        if k in opt:
            kwargs[k] = opt[k]
    return kwargs