import os
import shutil
import struct
from functools import lru_cache
from itertools import accumulate
import numpy as np
import torch
def end_document(self):
    self._doc_idx.append(len(self._sizes))