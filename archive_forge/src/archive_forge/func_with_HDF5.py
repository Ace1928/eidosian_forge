from .. import utils
from .._lazyload import h5py
from .._lazyload import tables
from decorator import decorator
@decorator
def with_HDF5(fun, *args, **kwargs):
    """Ensure that HDF5 is available to run the decorated function."""
    if not (utils._try_import('tables') or utils._try_import('h5py')):
        raise ModuleNotFoundError('Found neither tables nor h5py. Please install one of them with e.g. `pip install --user tables` or `pip install --user h5py`')
    return fun(*args, **kwargs)