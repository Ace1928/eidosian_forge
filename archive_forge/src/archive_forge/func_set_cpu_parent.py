import bisect
import itertools
import math
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.autograd import DeviceType
def set_cpu_parent(self, parent):
    """Set the immediate CPU parent of type FunctionEvent.

        One profiling FunctionEvent should have only one CPU parent such that
        the child's range interval is completely inside the parent's. We use
        this connection to determine the event is from top-level op or not.
        """
    assert self.device_type == DeviceType.CPU
    assert isinstance(parent, FunctionEvent)
    assert parent.device_type == DeviceType.CPU
    self.cpu_parent = parent