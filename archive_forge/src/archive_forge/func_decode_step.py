from __future__ import annotations
import itertools as it
from abc import abstractmethod
from collections import namedtuple
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
import torch
from flashlight.lib.text.decoder import (
from flashlight.lib.text.dictionary import (
from torchaudio.utils import download_asset
def decode_step(self, emissions: torch.FloatTensor):
    """Perform incremental decoding on top of the curent internal state.

        .. note::

           This method is required only when performing online decoding.
           It is not necessary when performing batch decoding with :py:meth:`__call__`.

        Args:
            emissions (torch.FloatTensor): CPU tensor of shape `(frame, num_tokens)` storing sequences of
                probability distribution over labels; output of acoustic model.

        Example:
            >>> decoder = torchaudio.models.decoder.ctc_decoder(...)
            >>> decoder.decode_begin()
            >>> decoder.decode_step(emission1)
            >>> decoder.decode_step(emission2)
            >>> decoder.decode_end()
            >>> result = decoder.get_final_hypothesis()
        """
    if emissions.dtype != torch.float32:
        raise ValueError('emissions must be float32.')
    if not emissions.is_cpu:
        raise RuntimeError('emissions must be a CPU tensor.')
    if not emissions.is_contiguous():
        raise RuntimeError('emissions must be contiguous.')
    if emissions.ndim != 2:
        raise RuntimeError(f'emissions must be 2D. Found {emissions.shape}')
    T, N = emissions.size()
    self.decoder.decode_step(emissions.data_ptr(), T, N)