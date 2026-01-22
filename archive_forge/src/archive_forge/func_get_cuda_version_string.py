import dataclasses
from typing import List, Optional, Tuple
import torch
def get_cuda_version_string() -> str:
    major, minor = get_cuda_version_tuple()
    return f'{major}{minor}'