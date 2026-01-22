from __future__ import annotations
import itertools as it
from abc import abstractmethod
from collections import namedtuple
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
import torch
from flashlight.lib.text.decoder import (
from flashlight.lib.text.dictionary import (
from torchaudio.utils import download_asset
def decode_begin(self):
    """Initialize the internal state of the decoder.

        See :py:meth:`decode_step` for the usage.

        .. note::

           This method is required only when performing online decoding.
           It is not necessary when performing batch decoding with :py:meth:`__call__`.
        """
    self.decoder.decode_begin()