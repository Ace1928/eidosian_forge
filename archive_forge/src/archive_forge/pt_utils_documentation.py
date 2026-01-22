import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from ..utils.generic import ModelOutput
Subiterator None means we haven't started a `preprocess` iterator. so start it