from tqdm import tqdm, tqdm_notebook
from collections import OrderedDict
import time
def close_tqdm_bar(self, bar):
    """Close and erase the tqdm bar"""
    self.tqdm_bars[bar].close()
    if not self.notebook:
        self.tqdm_bars[bar] = None