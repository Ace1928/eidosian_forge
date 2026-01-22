from tqdm import tqdm, tqdm_notebook
from collections import OrderedDict
import time
def bar_is_logged(self, bar):
    if not self.logged_bars:
        return False
    elif self.logged_bars == 'all':
        return True
    else:
        return bar in self.logged_bars