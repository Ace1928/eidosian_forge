import collections
import contextlib
import dataclasses
import os
import shutil
import tempfile
import textwrap
import time
from typing import cast, Any, DefaultDict, Dict, Iterable, Iterator, List, Optional, Tuple
import uuid
import torch
def select_unit(t: float) -> Tuple[str, float]:
    """Determine how to scale times for O(1) magnitude.

    This utility is used to format numbers for human consumption.
    """
    time_unit = {-3: 'ns', -2: 'us', -1: 'ms'}.get(int(torch.tensor(t).log10().item() // 3), 's')
    time_scale = {'ns': 1e-09, 'us': 1e-06, 'ms': 0.001, 's': 1}[time_unit]
    return (time_unit, time_scale)