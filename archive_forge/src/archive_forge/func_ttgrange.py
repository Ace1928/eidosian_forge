from os import getenv
from warnings import warn
from requests import Session
from ..auto import tqdm as tqdm_auto
from ..std import TqdmWarning
from .utils_worker import MonoWorker
def ttgrange(*args, **kwargs):
    """Shortcut for `tqdm.contrib.telegram.tqdm(range(*args), **kwargs)`."""
    return tqdm_telegram(range(*args), **kwargs)