import numpy as np
import os
from os import path
import sys
import pytest
from ctypes import c_longlong, c_double, c_float, c_int, cast, pointer, POINTER
from numpy.testing import assert_array_max_ulp
from numpy.testing._private.utils import _glibc_older_than
from numpy.core._multiarray_umath import __cpu_features__
@platform_skip
def test_validate_transcendentals(self):
    with np.errstate(all='ignore'):
        data_dir = path.join(path.dirname(__file__), 'data')
        files = os.listdir(data_dir)
        files = list(filter(lambda f: f.endswith('.csv'), files))
        for filename in files:
            filepath = path.join(data_dir, filename)
            with open(filepath) as fid:
                file_without_comments = (r for r in fid if not r[0] in ('$', '#'))
                data = np.genfromtxt(file_without_comments, dtype=('|S39', '|S39', '|S39', int), names=('type', 'input', 'output', 'ulperr'), delimiter=',', skip_header=1)
                npname = path.splitext(filename)[0].split('-')[3]
                npfunc = getattr(np, npname)
                for datatype in np.unique(data['type']):
                    data_subset = data[data['type'] == datatype]
                    inval = np.array(str_to_float(data_subset['input'].astype(str), data_subset['type'].astype(str)), dtype=eval(datatype))
                    outval = np.array(str_to_float(data_subset['output'].astype(str), data_subset['type'].astype(str)), dtype=eval(datatype))
                    perm = np.random.permutation(len(inval))
                    inval = inval[perm]
                    outval = outval[perm]
                    maxulperr = data_subset['ulperr'].max()
                    assert_array_max_ulp(npfunc(inval), outval, maxulperr)