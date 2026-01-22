import operator
import warnings
import dask
from dask import core
from dask.core import istask
from dask.dataframe.core import _concat
from dask.dataframe.optimize import optimize
from dask.dataframe.shuffle import shuffle_group
from dask.highlevelgraph import HighLevelGraph
from .scheduler import MultipleReturnFunc, multiple_return_get
def dataframe_optimize(dsk, keys, **kwargs):
    warnings.warn(f'Custom dataframe shuffle optimization only works on dask>=2020.12.0, you are on version {dask.__version__}, please upgrade Dask.Falling back to default dataframe optimizer.')
    return optimize(dsk, keys, **kwargs)