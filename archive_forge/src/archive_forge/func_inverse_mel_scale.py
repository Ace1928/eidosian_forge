import math
from typing import Tuple
import torch
import torchaudio
from torch import Tensor
def inverse_mel_scale(mel_freq: Tensor) -> Tensor:
    return 700.0 * ((mel_freq / 1127.0).exp() - 1.0)