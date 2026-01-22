from io import BytesIO
from itertools import product
import random
from typing import Any, List
import torch
def format_with_label(label: str, value: Any) -> str:
    if isinstance(value, bool):
        formatted = 'T' if value else 'F'
    elif isinstance(value, (list, tuple)) and all((isinstance(v, bool) for v in value)):
        formatted = ''.join(('T' if b else 'F' for b in value))
    else:
        formatted = str(value)
    return f'{label}={formatted}'