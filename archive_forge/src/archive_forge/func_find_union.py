import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.imports import _NLTK_AVAILABLE
def find_union(lcs_tables: Sequence[Sequence[int]]) -> Sequence[int]:
    """Find union LCS given a list of LCS."""
    return sorted(set().union(*lcs_tables))