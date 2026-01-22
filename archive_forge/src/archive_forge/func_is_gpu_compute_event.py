import contextlib
import time
import os
import json
import torch
from torch.profiler import profile, ProfilerActivity
def is_gpu_compute_event(event):
    global gpu_pids
    return 'pid' in event and event['pid'] in gpu_pids and ('ph' in event) and (event['ph'] == 'X')