import math
import os
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchmetrics.utilities.data import _cumsum
from torchmetrics.utilities.imports import _TQDM_AVAILABLE, _TRANSFORMERS_GREATER_EQUAL_4_4
Initialize the dataset class.

        Args:
            input_ids: Input indexes
            attention_mask: Attention mask
            idf:
                An indication of whether calculate token inverse document frequencies to weight the model embeddings.
            tokens_idf: Inverse document frequencies (these should be calculated on reference sentences).

        