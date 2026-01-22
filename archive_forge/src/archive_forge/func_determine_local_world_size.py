import logging
import os
import sys
import uuid
from argparse import REMAINDER, ArgumentParser
from typing import Callable, List, Tuple, Union
import torch
from torch.distributed.argparse_util import check_env, env
from torch.distributed.elastic.multiprocessing import Std
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.elastic.rendezvous.utils import _parse_rendezvous_config
from torch.distributed.elastic.utils import macros
from torch.distributed.elastic.utils.logging import get_logger
from torch.distributed.launcher.api import LaunchConfig, elastic_launch
from torch.utils.backend_registration import _get_custom_mod_func
def determine_local_world_size(nproc_per_node: str):
    try:
        logging.info('Using nproc_per_node=%s.', nproc_per_node)
        return int(nproc_per_node)
    except ValueError as e:
        if nproc_per_node == 'cpu':
            num_proc = os.cpu_count()
            device_type = 'cpu'
        elif nproc_per_node == 'gpu':
            if not torch.cuda.is_available():
                raise ValueError('Cuda is not available.') from e
            device_type = 'gpu'
            num_proc = torch.cuda.device_count()
        elif nproc_per_node == torch._C._get_privateuse1_backend_name():
            if not _get_custom_mod_func('is_available')():
                raise ValueError(f'{nproc_per_node} is not available.') from e
            device_type = nproc_per_node
            num_proc = _get_custom_mod_func('device_count')()
        elif nproc_per_node == 'auto':
            if torch.cuda.is_available():
                num_proc = torch.cuda.device_count()
                device_type = 'gpu'
            elif hasattr(torch, torch._C._get_privateuse1_backend_name()) and _get_custom_mod_func('is_available')():
                num_proc = _get_custom_mod_func('device_count')()
                device_type = torch._C._get_privateuse1_backend_name()
            else:
                num_proc = os.cpu_count()
                device_type = 'cpu'
        else:
            raise ValueError(f'Unsupported nproc_per_node value: {nproc_per_node}') from e
        log.info('Using nproc_per_node=%s, setting to %s since the instance has %s %s', nproc_per_node, num_proc, os.cpu_count(), device_type)
        return num_proc