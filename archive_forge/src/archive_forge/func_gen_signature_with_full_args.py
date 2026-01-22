from collections import defaultdict
from pathlib import Path
from typing import Sequence, Union
from dataclasses import dataclass
def gen_signature_with_full_args(m):
    return ', '.join([f'{ty} {arg}' for ty, arg in zip(m.arg_ctypes, m.arg_names)])