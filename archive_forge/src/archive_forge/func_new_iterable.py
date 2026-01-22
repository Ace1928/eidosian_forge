from tqdm import tqdm, tqdm_notebook
from collections import OrderedDict
import time
def new_iterable():
    last_time = time.time()
    i = 0
    for i, it in enumerate(iterable):
        now_time = time.time()
        if i == 0 or now_time - last_time > self.min_time_interval:
            if bar_message is not None:
                self(**{bar + '__message': bar_message(it)})
            self(**{bar + '__index': i})
            last_time = now_time
        yield it
    if self.bars[bar]['index'] != i:
        self(**{bar + '__index': i})
    self(**{bar + '__index': i + 1})