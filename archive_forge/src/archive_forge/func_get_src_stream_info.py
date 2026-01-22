from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Tuple, TypeVar, Union
import torch
import torio
from torch.utils._pytree import tree_map
def get_src_stream_info(self, i: int) -> InputStreamTypes:
    """Get the metadata of source stream

        Args:
            i (int): Stream index.
        Returns:
            InputStreamTypes:
                Information about the source stream.
                If the source stream is audio type, then
                :class:`~torio.io._stream_reader.SourceAudioStream` is returned.
                If it is video type, then
                :class:`~torio.io._stream_reader.SourceVideoStream` is returned.
                Otherwise :class:`~torio.io._stream_reader.SourceStream` class is returned.
        """
    return _parse_si(self._be.get_src_stream_info(i))