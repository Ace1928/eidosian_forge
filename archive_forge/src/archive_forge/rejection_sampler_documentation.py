from typing import Tuple, Optional
from functools import cached_property
import torch
import torch.nn as nn
import torch.jit
Format output. Returns a matrix of token ids. When
        a token is rejected via rejection sampling, all subsequent
        token ids are set to -1 for the sequence.

        shape = [batch_size, k + num_bonus_tokens]
        