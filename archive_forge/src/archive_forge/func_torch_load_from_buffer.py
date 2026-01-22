from io import BytesIO
from itertools import product
import random
from typing import Any, List
import torch
def torch_load_from_buffer(buffer):
    buffer.seek(0)
    obj = torch.load(buffer)
    buffer.seek(0)
    return obj