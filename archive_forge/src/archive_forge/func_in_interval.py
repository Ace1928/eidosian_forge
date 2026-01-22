import bisect
import itertools
import math
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.autograd import DeviceType
def in_interval(self, start_us, end_us):
    start_idx = bisect.bisect_left(self._start_uses, start_us)
    end_idx = bisect.bisect_right(self._start_uses, end_us)
    for i in range(start_idx, end_idx):
        yield self._mem_records[self._indices[i]]