import functools
import torch
import torch.library
import torchvision.extension  # noqa: F401
@functools.lru_cache(None)
def get_meta_lib():
    return torch.library.Library('torchvision', 'IMPL', 'Meta')