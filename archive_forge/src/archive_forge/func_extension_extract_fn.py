import io
import json
import os.path
import pickle
import tempfile
import torch
from torch.utils.data.datapipes.utils.common import StreamWrapper
def extension_extract_fn(pathname):
    ext = os.path.splitext(pathname)[1]
    if ext:
        ext = ext[1:]
    return ext