import logging
import math
import time
from golden_configs.lm_wikitext2 import MOE as MOEConfig
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import utils
Benchmark a given model using a single process and multiple devices.