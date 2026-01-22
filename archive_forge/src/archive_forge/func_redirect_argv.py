import argparse
import cProfile
import pstats
import sys
import os
from typing import Dict
import torch
from torch.autograd import profiler
from torch.utils.collect_env import get_env_info
def redirect_argv(new_argv):
    sys.argv[:] = new_argv[:]