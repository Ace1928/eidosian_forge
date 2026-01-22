from collections import defaultdict
from pathlib import Path
from typing import Sequence, Union
from dataclasses import dataclass
def gen_signature(m):
    arg_types = [ty for ty, hint in zip(m.arg_ctypes, m.sizes) if hint != 1]
    arg_names = [arg for arg, hint in zip(m.arg_names, m.sizes) if hint != 1]
    sig = ', '.join([f'{ty} {arg}' for ty, arg in zip(arg_types, arg_names)])
    return sig