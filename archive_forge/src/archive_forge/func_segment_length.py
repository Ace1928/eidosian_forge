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
def segment_length(self) -> int:
    """Number of frames in segment in input expected by model.

        :type: int
        """
    return self._segment_length