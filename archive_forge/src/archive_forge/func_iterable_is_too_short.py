from tqdm import tqdm, tqdm_notebook
from collections import OrderedDict
import time
def iterable_is_too_short(self, iterable):
    length = len(iterable) if hasattr(iterable, '__len__') else None
    return length is not None and length < self.ignore_bars_under