import argparse
from collections import defaultdict
from functools import reduce
import gc
import logging
import math
import operator
import time
from datasets.wikitext2_data import get_real_dataloaders as get_real_wikitext2_dataloaders
from datasets.wikitext2_data import get_synthetic_dataloaders as get_synthetic_wikitext2_dataloaders
from models import transformer_lm
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from benchmarks.golden_configs.lm_wikitext2 import FSDP as lm_wikitext2
from fairscale.nn import auto_wrap, default_auto_wrap_policy, enable_wrap
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
def benchmark_language_model(model_config, model, benchmark_config, model_specs, args):
    golden_config = get_golden_config(args.model_name, args)
    epoch = benchmark_config['epochs']
    start_time = time.time()
    if dist.get_rank() == 0:
        print('-' * 110)
        print('| start of epoch {:1d}'.format(epoch))
        print('-' * 110)
    wps, loss = train(model_config, model, benchmark_config, model_specs, args)
    elapsed_time = time.time() - start_time
    if dist.get_rank() == 0:
        print('-' * 110)
        print('| end of epoch {:1d} | time: {:5.2f}s | train loss {:5.2f} '.format(epoch, elapsed_time, loss))
        print('-' * 110)
        print('Throughput(wps) is {:.2f}.'.format(wps))
    print('Peak allocated bytes on cuda:{}: {:4f}GB'.format(dist.get_rank(), torch.cuda.memory_stats(dist.get_rank())['allocated_bytes.all.peak'] / 2 ** 30))
    verify_lm_run(wps, golden_config, args)