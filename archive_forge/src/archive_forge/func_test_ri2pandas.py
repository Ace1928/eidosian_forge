import math
from collections import OrderedDict
from datetime import datetime
import pytest
from rpy2 import rinterface
from rpy2 import robjects
from rpy2.robjects import vectors
from rpy2.robjects import conversion
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
@pytest.mark.parametrize('colnames', (('w', 'x', 'y', 'z'), ('w', 'x', 'y', 'x')))
@pytest.mark.parametrize('r_rownames,py_rownames', (('NULL', ('1', '2')), ('c("a", "b")', ('a', 'b'))))
def test_ri2pandas(self, r_rownames, py_rownames, colnames):
    rdataf = robjects.r(f'data.frame({colnames[0]}=1:2,            {colnames[1]}=1:2,            {colnames[2]}=I(c("a", "b")),            {colnames[3]}=factor(c("a", "b")),            row.names={r_rownames},             check.names=FALSE)')
    with localconverter(default_converter + rpyp.converter) as cv:
        pandas_df = robjects.conversion.converter_ctx.get().rpy2py(rdataf)
    assert isinstance(pandas_df, pandas.DataFrame)
    assert colnames == tuple(pandas_df.keys())
    assert pandas_df['w'].dtype in (numpy.dtype('int32'), numpy.dtype('int64'))
    assert pandas_df['y'].dtype == numpy.dtype('O')
    if 'z' in colnames:
        assert isinstance(pandas_df['z'].dtype, pandas.api.types.CategoricalDtype)
    assert tuple(pandas_df.index) == py_rownames