from __future__ import division, print_function, absolute_import
from datetime import datetime
from tempfile import NamedTemporaryFile
import pytest
import petl as etl
from petl.io.xls import fromxls, toxls
from petl.test.helpers import ieq
def test_toxls_date():
    expect = (('foo', 'bar'), (u'é', datetime(2012, 1, 1)), (u'éé', datetime(2013, 2, 22)))
    f = NamedTemporaryFile(delete=False)
    f.close()
    toxls(expect, f.name, 'Sheet1', styles={'bar': xlwt.easyxf(num_format_str='DD/MM/YYYY')})
    actual = fromxls(f.name, 'Sheet1')
    ieq(expect, actual)