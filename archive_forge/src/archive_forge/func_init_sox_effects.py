import os
from typing import List, Optional, Tuple
import torch
import torchaudio
from torchaudio._internal.module_utils import deprecated
from torchaudio.utils.sox_utils import list_effects
@deprecated('Please remove the call. This function is called automatically.')
def init_sox_effects():
    """Initialize resources required to use sox effects.

    Note:
        You do not need to call this function manually. It is called automatically.

    Once initialized, you do not need to call this function again across the multiple uses of
    sox effects though it is safe to do so as long as :func:`shutdown_sox_effects` is not called yet.
    Once :func:`shutdown_sox_effects` is called, you can no longer use SoX effects and initializing
    again will result in error.
    """
    pass