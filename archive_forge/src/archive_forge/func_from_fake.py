import functools
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
@staticmethod
def from_fake(fake) -> 'FakeTensorMeta':
    return FakeTensorMeta(fake.size(), fake.stride(), fake.storage_offset(), fake.is_nested)