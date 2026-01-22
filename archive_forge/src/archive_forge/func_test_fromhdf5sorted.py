from __future__ import division, print_function, absolute_import
from itertools import chain
from tempfile import NamedTemporaryFile
import pytest
from petl.test.helpers import ieq
from petl.transform.sorts import sort
import petl as etl
from petl.io.pytables import fromhdf5, fromhdf5sorted, tohdf5, appendhdf5
def test_fromhdf5sorted():
    f = NamedTemporaryFile()
    h5file = tables.open_file(f.name, mode='w', title='Test file')
    h5file.create_group('/', 'testgroup', 'Test Group')
    h5table = h5file.create_table('/testgroup', 'testtable', FooBar, 'Test Table')
    table1 = (('foo', 'bar'), (3, b'asdfgh'), (2, b'qwerty'), (1, b'zxcvbn'))
    for row in table1[1:]:
        for i, f in enumerate(table1[0]):
            h5table.row[f] = row[i]
        h5table.row.append()
    h5table.cols.foo.create_csindex()
    h5file.flush()
    table2 = fromhdf5sorted(h5table, sortby='foo')
    ieq(sort(table1, 'foo'), table2)
    ieq(sort(table1, 'foo'), table2)
    h5file.close()