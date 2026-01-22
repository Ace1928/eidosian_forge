from io import BytesIO
from itertools import product
import random
from typing import Any, List
import torch
def torch_save_to_buffer(obj):
    buffer = BytesIO()
    torch.save(obj, buffer)
    buffer.seek(0)
    return buffer