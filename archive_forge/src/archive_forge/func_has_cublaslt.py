import dataclasses
from typing import List, Optional, Tuple
import torch
@property
def has_cublaslt(self) -> bool:
    return self.highest_compute_capability >= (7, 5)