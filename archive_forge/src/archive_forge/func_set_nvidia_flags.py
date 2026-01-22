import logging
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Union
import torch
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.accelerators import _AcceleratorRegistry
from lightning_fabric.accelerators.cuda import _check_cuda_matmul_precision, _clear_cuda_memory, num_cuda_devices
from lightning_fabric.utilities.device_parser import _parse_gpu_ids
from lightning_fabric.utilities.types import _DEVICE
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities.exceptions import MisconfigurationException
@staticmethod
def set_nvidia_flags(local_rank: int) -> None:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    all_gpu_ids = ','.join((str(x) for x in range(num_cuda_devices())))
    devices = os.getenv('CUDA_VISIBLE_DEVICES', all_gpu_ids)
    _log.info(f'LOCAL_RANK: {local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]')