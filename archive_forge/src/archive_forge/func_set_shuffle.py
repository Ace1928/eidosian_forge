import random
import torch
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe
from typing import Iterator, List, Optional, TypeVar
def set_shuffle(self, shuffle=True):
    self._enabled = shuffle
    return self