import abc
import functools
import json
import os
import signal
import socket
import time
import traceback
import warnings
from contextlib import closing
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch.distributed.elastic.rendezvous as rdzv
import torch.distributed.elastic.utils.store as store_util
from torch.distributed import Store
from torch.distributed.elastic.events import Event, EventSource, record
from torch.distributed.elastic.metrics import prof, put_metric
from torch.distributed.elastic.multiprocessing import (
from torch.distributed.elastic.utils.logging import get_logger
@staticmethod
def find_role_boundaries(roles_infos: List, role: str) -> Tuple[int, int]:
    start_idx, end_idx = (-1, -1)
    for idx, role_info in enumerate(roles_infos):
        if role_info.role == role:
            if start_idx == -1:
                start_idx = idx
            end_idx = idx
    return (start_idx, end_idx)