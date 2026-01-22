import argparse
import cProfile
import pstats
import sys
import os
from typing import Dict
import torch
from torch.autograd import profiler
from torch.utils.collect_env import get_env_info
def cpu_time_total(autograd_prof):
    return sum([event.cpu_time_total for event in autograd_prof.function_events])