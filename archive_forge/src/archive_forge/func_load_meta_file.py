import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple
import torch
from .folder import ImageFolder
from .utils import check_integrity, extract_archive, verify_str_arg
def load_meta_file(root: str, file: Optional[str]=None) -> Tuple[Dict[str, str], List[str]]:
    if file is None:
        file = META_FILE
    file = os.path.join(root, file)
    if check_integrity(file):
        return torch.load(file)
    else:
        msg = 'The meta file {} is not present in the root directory or is corrupted. This file is automatically created by the ImageNet dataset.'
        raise RuntimeError(msg.format(file, root))