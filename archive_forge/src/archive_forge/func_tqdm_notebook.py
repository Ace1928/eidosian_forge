from ._monitor import TMonitor, TqdmSynchronisationWarning
from ._tqdm_pandas import tqdm_pandas
from .cli import main  # TODO: remove in v5.0.0
from .gui import tqdm as tqdm_gui  # TODO: remove in v5.0.0
from .gui import trange as tgrange  # TODO: remove in v5.0.0
from .std import (
from .version import __version__
def tqdm_notebook(*args, **kwargs):
    """See tqdm.notebook.tqdm for full documentation"""
    from warnings import warn
    from .notebook import tqdm as _tqdm_notebook
    warn('This function will be removed in tqdm==5.0.0\nPlease use `tqdm.notebook.tqdm` instead of `tqdm.tqdm_notebook`', TqdmDeprecationWarning, stacklevel=2)
    return _tqdm_notebook(*args, **kwargs)