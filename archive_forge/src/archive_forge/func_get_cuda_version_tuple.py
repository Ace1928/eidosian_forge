import dataclasses
from typing import List, Optional, Tuple
import torch
def get_cuda_version_tuple() -> Tuple[int, int]:
    major, minor = map(int, torch.version.cuda.split('.'))
    return (major, minor)