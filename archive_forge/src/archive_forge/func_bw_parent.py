import bisect
import itertools
import math
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.autograd import DeviceType
def bw_parent(evt):
    if evt is None:
        return None
    elif evt.scope == 1:
        return evt
    else:
        return bw_parent(evt.cpu_parent)