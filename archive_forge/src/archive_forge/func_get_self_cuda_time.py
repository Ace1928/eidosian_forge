import dataclasses
import tempfile
from collections import defaultdict
import torch
from torch.autograd import DeviceType
from .utils import create_bandwidth_info_str, do_bench, get_num_bytes
def get_self_cuda_time(ev):
    """
        ev.self_cuda_time_total is in microsecond. Convert to millisecond.
        """
    return ev.self_cuda_time_total / 1000 / nruns