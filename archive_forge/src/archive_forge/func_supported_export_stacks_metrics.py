import bisect
import itertools
import math
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.autograd import DeviceType
def supported_export_stacks_metrics(self):
    return ['self_cpu_time_total', 'self_cuda_time_total', 'self_privateuse1_time_total']