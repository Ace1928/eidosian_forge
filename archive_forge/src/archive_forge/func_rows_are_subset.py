from torch.ao.pruning import BaseSparsifier
import torch
import torch.nn.functional as F
from torch import nn
def rows_are_subset(subset_tensor, superset_tensor) -> bool:
    """
    Checks to see if all rows in subset tensor are present in the superset tensor
    """
    i = 0
    for row in subset_tensor:
        while i < len(superset_tensor):
            if not torch.equal(row, superset_tensor[i]):
                i += 1
            else:
                break
        else:
            return False
    return True