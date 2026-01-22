import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Tuple
import torch
import torchaudio
from torchaudio._internal import module_utils
from torchaudio.models import emformer_rnnt_base, RNNT, RNNTBeamSearch
@property
def hop_length(self) -> int:
    """Number of samples between successive frames in input expected by model.

        :type: int
        """
    return self._hop_length