import logging
from os import getenv
from ..auto import tqdm as tqdm_auto
from .utils_worker import MonoWorker
def tdrange(*args, **kwargs):
    """Shortcut for `tqdm.contrib.discord.tqdm(range(*args), **kwargs)`."""
    return tqdm_discord(range(*args), **kwargs)