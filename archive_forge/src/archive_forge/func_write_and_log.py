import gc
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import types
import weakref
import json
from tempfile import NamedTemporaryFile
import torch
from torch.cuda._memory_viz import _frames_fmt, _block_extra
import atexit
import logging
def write_and_log(html):
    with NamedTemporaryFile('w', suffix='.html', delete=False) as f:
        f.write(html)
        logger.warning('Reference cycle includes a CUDA Tensor see visualization of cycle %s', f.name)