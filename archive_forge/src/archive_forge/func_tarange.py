import asyncio
from sys import version_info
from .std import tqdm as std_tqdm
def tarange(*args, **kwargs):
    """
    A shortcut for `tqdm.asyncio.tqdm(range(*args), **kwargs)`.
    """
    return tqdm_asyncio(range(*args), **kwargs)