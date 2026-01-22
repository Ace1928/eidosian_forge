import re
import sys
from html import escape
from weakref import proxy
from .std import tqdm as std_tqdm
def tnrange(*args, **kwargs):
    """Shortcut for `tqdm.notebook.tqdm(range(*args), **kwargs)`."""
    return tqdm_notebook(range(*args), **kwargs)