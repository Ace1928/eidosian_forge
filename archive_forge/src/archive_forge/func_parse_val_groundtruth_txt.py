import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple
import torch
from .folder import ImageFolder
from .utils import check_integrity, extract_archive, verify_str_arg
def parse_val_groundtruth_txt(devkit_root: str) -> List[int]:
    file = os.path.join(devkit_root, 'data', 'ILSVRC2012_validation_ground_truth.txt')
    with open(file) as txtfh:
        val_idcs = txtfh.readlines()
    return [int(val_idx) for val_idx in val_idcs]