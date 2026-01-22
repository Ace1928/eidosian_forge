import contextlib
import time
import os
import json
import torch
from torch.profiler import profile, ProfilerActivity
def is_mm_conv_event(event):
    return 'name' in event and ('gemm' in event['name'] or 'conv' in event['name'] or 'cutlass' in event['name'] or ('wgrad' in event['name']))