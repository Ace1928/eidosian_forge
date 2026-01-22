from enum import Enum
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from xformers import _is_triton_available
from collections import namedtuple

    See DeepNet_.

    Returns alpha and beta depending on the number of encoder and decoder layers,
    first tuple is for the encoder and second for the decoder

    .. _DeepNet: https://arxiv.org/pdf/2203.00555v1.pdf
    