from typing import Dict
import numpy as np
import torch as th
import torch.nn as nn
from parlai.utils.torch import neginf
from parlai.agents.transformer.modules import TransformerGeneratorModel
def reorder_encoder_states(self, encoder_out, indices):
    enc, mask, ckattn = encoder_out
    if not th.is_tensor(indices):
        indices = th.LongTensor(indices).to(enc.device)
    enc = th.index_select(enc, 0, indices)
    mask = th.index_select(mask, 0, indices)
    ckattn = th.index_select(ckattn, 0, indices)
    return (enc, mask, ckattn)