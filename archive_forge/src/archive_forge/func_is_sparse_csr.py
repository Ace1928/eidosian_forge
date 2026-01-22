import warnings
import torch
from torch.overrides import get_default_nowrap_functions
def is_sparse_csr(self):
    return self.layout == torch.sparse_csr