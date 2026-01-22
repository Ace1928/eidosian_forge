import enum
import functools
import inspect
import io
import logging
import sys
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, TypeVar
import torch
import torch.utils._cuda_trace as cuda_trace
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
def update_seq_num(self, stream: StreamId, seq_num: SeqNum) -> None:
    self._ensure_stream_exists(stream)
    self.current_sync_states[stream][stream] = seq_num