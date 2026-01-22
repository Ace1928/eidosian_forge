import math
from typing import Dict, Tuple, Optional, Union
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from parlai.core.torch_generator_agent import TorchGeneratorModel
from parlai.utils.misc import warn_once
from parlai.utils.torch import neginf, PipelineHelper
def encode_context_memory(self, context_w, memories_w, context_segments=None):
    """
        Encode the context and memories.
        """
    if context_w is None:
        return (None, None)
    context_h = self.context_encoder(context_w, segments=context_segments)
    if memories_w is None:
        return ([], context_h)
    bsz = memories_w.size(0)
    memories_w = memories_w.view(-1, memories_w.size(-1))
    memories_h = self.memory_transformer(memories_w)
    memories_h = memories_h.view(bsz, -1, memories_h.size(-1))
    context_h = context_h.unsqueeze(1)
    context_h, weights = self.attender(context_h, memories_h)
    return (weights, context_h)