from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Tuple, TypeVar, Union
import torch
import torio
from torch.utils._pytree import tree_map
def fill_buffer(self, timeout: Optional[float]=None, backoff: float=10.0) -> int:
    """Keep processing packets until all buffers have at least one chunk

        Arguments:
            timeout (float or None, optional): See
                :py:func:`~StreamingMediaDecoder.process_packet`. (Default: ``None``)

            backoff (float, optional): See
                :py:func:`~StreamingMediaDecoder.process_packet`. (Default: ``10.0``)

        Returns:
            int:
                ``0``
                Packets are processed properly and buffers are
                ready to be popped once.

                ``1``
                The streamer reached EOF. All the output stream processors
                flushed the pending frames. The caller should stop calling
                this method.
        """
    return self._be.fill_buffer(timeout, backoff)