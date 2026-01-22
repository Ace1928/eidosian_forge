import bisect
import itertools
import math
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.autograd import DeviceType
def trim_path(path, src_column_width):
    if len(path) > src_column_width:
        offset = len(path) - src_column_width
        path = path[offset:]
        if len(path) > 3:
            path = '...' + path[3:]
    return path