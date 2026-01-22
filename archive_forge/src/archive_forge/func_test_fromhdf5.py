from __future__ import division, print_function, absolute_import
from itertools import chain
from tempfile import NamedTemporaryFile
import pytest
from petl.test.helpers import ieq
from petl.transform.sorts import sort
import petl as etl
from petl.io.pytables import fromhdf5, fromhdf5sorted, tohdf5, appendhdf5
def test_fromhdf5():
    f = NamedTemporaryFile()
    h5file = tables.open_file(f.name, mode='w', title='Test file')
    h5file.create_group('/', 'testgroup', 'Test Group')
    h5table = h5file.create_table('/testgroup', 'testtable', FooBar, 'Test Table')
    table1 = (('foo', 'bar'), (1, b'asdfgh'), (2, b'qwerty'), (3, b'zxcvbn'))
    for row in table1[1:]:
        for i, fld in enumerate(table1[0]):
            h5table.row[fld] = row[i]
        h5table.row.append()
    h5file.flush()
    h5file.close()
    table2a = fromhdf5(f.name, '/testgroup', 'testtable')
    ieq(table1, table2a)
    ieq(table1, table2a)
    table2b = fromhdf5(f.name, '/testgroup/testtable')
    ieq(table1, table2b)
    ieq(table1, table2b)
    h5file = tables.open_file(f.name)
    table3 = fromhdf5(h5file, '/testgroup/testtable')
    ieq(table1, table3)
    h5tbl = h5file.get_node('/testgroup/testtable')
    table4 = fromhdf5(h5tbl)
    ieq(table1, table4)
    table5 = fromhdf5(h5tbl, condition='(foo < 3)')
    ieq(table1[:3], table5)
    h5file.close()