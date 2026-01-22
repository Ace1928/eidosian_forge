import sys
from contextlib import nullcontext
import numpy as np
import pytest
from packaging.version import Version
from ..casting import (
from ..testing import suppress_warnings
def test_type_info():
    for dtt in sctypes['int'] + sctypes['uint']:
        info = np.iinfo(dtt)
        infod = type_info(dtt)
        assert infod == dict(min=info.min, max=info.max, nexp=None, nmant=None, minexp=None, maxexp=None, width=np.dtype(dtt).itemsize)
        assert infod['min'].dtype.type == dtt
        assert infod['max'].dtype.type == dtt
    for dtt in IEEE_floats + [np.complex64, np.complex64]:
        infod = type_info(dtt)
        assert dtt2dict(dtt) == infod
        assert infod['min'].dtype.type == dtt
        assert infod['max'].dtype.type == dtt
    ld_dict = dtt2dict(np.longdouble)
    dbl_dict = dtt2dict(np.float64)
    infod = type_info(np.longdouble)
    vals = tuple((ld_dict[k] for k in ('nmant', 'nexp', 'width')))
    if vals in ((52, 11, 8), (63, 15, 12), (63, 15, 16), (112, 15, 16), (106, 11, 16)):
        pass
    elif vals == (105, 11, 16):
        ld_dict.update({k: infod[k] for k in ('min', 'max')})
    elif vals == (1, 1, 16):
        ld_dict = dbl_dict.copy()
        ld_dict.update(dict(nmant=106, width=16))
    elif vals == (52, 15, 12):
        width = ld_dict['width']
        ld_dict = dbl_dict.copy()
        ld_dict['width'] = width
    else:
        raise ValueError(f'Unexpected float type {np.longdouble} to test')
    assert ld_dict == infod