import contextlib
import time
import os
import json
import torch
from torch.profiler import profile, ProfilerActivity
def get_sorted_gpu_events(events):
    sorted_gpu_events = []
    for event in events:
        if not is_gpu_compute_event(event):
            continue
        sorted_gpu_events.append(event)
    return sorted(sorted_gpu_events, key=lambda x: x['ts'])