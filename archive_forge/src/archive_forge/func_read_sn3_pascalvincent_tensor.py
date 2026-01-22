import codecs
import os
import os.path
import shutil
import string
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import URLError
import numpy as np
import torch
from PIL import Image
from .utils import _flip_byte_order, check_integrity, download_and_extract_archive, extract_archive, verify_str_arg
from .vision import VisionDataset
def read_sn3_pascalvincent_tensor(path: str, strict: bool=True) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
    Argument may be a filename, compressed filename, or file object.
    """
    with open(path, 'rb') as f:
        data = f.read()
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    torch_type = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1):4 * (i + 2)]) for i in range(nd)]
    parsed = torch.frombuffer(bytearray(data), dtype=torch_type, offset=4 * (nd + 1))
    if sys.byteorder == 'little' and parsed.element_size() > 1:
        parsed = _flip_byte_order(parsed)
    assert parsed.shape[0] == np.prod(s) or not strict
    return parsed.view(*s)