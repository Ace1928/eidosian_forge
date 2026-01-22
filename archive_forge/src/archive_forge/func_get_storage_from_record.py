import os.path
from glob import glob
from typing import cast
import torch
from torch.types import Storage
def get_storage_from_record(self, name, numel, dtype):
    filename = f'{self.directory}/{name}'
    nbytes = torch._utils._element_size(dtype) * numel
    storage = cast(Storage, torch.UntypedStorage)
    return _HasStorage(storage.from_file(filename=filename, nbytes=nbytes))