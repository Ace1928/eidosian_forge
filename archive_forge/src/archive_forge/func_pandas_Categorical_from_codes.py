import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def pandas_Categorical_from_codes(codes, categories):
    assert have_pandas_categorical
    codes = np.asarray(codes)
    if hasattr(pandas.Categorical, 'from_codes'):
        return pandas.Categorical.from_codes(codes, categories)
    else:
        return pandas.Categorical(codes, categories)