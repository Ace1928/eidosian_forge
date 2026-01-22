import argparse
import fnmatch
import re
import shutil
import sys
import os
from . import constants
from .cuda_to_hip_mappings import CUDA_TO_HIP_MAPPINGS
from .cuda_to_hip_mappings import MATH_TRANSPILATIONS
from typing import Dict, List, Iterator, Optional
from collections.abc import Mapping, Iterable
from enum import Enum
def is_pytorch_file(rel_filepath):
    assert not os.path.isabs(rel_filepath)
    if rel_filepath.startswith('aten/'):
        if rel_filepath.startswith('aten/src/ATen/core/'):
            return False
        return True
    if rel_filepath.startswith('torch/'):
        return True
    if rel_filepath.startswith('third_party/nvfuser/'):
        return True
    if rel_filepath.startswith('tools/autograd/templates/'):
        return True
    return False