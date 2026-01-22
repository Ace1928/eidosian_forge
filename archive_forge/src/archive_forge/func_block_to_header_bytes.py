from functools import partial
import pickle
import numpy as np
import pandas as pd
from pandas.core.internals import create_block_manager_from_blocks, make_block
from . import numpy as pnp
from .core import Interface
from .encode import Encode
from .utils import extend, framesplit, frame
def block_to_header_bytes(block):
    values = block.values
    if isinstance(values, pd.Categorical):
        extension = ('categorical_type', (values.ordered, values.categories))
        values = values.codes
    elif isinstance(block, pd.DatetimeTZDtype):
        extension = ('datetime64_tz_type', (block.values.tzinfo,))
        values = values.view('i8')
    elif is_extension_array_dtype(block.dtype) or is_extension_array(values):
        extension = ('other', ())
    else:
        extension = ('numpy_type', ())
    header = (block.mgr_locs.as_array, values.dtype, values.shape, extension)
    if extension == ('other', ()):
        bytes = pickle.dumps(values)
    else:
        bytes = pnp.compress(pnp.serialize(values), values.dtype)
    return (header, bytes)