import pytest
import decimal
import datetime
import pyarrow as pa
from pyarrow import fs
from pyarrow.tests import util
def test_column_selection(tempdir):
    from pyarrow import orc
    inner = pa.field('inner', pa.int64())
    middle = pa.field('middle', pa.struct([inner]))
    fields = [pa.field('basic', pa.int32()), pa.field('list', pa.list_(pa.field('item', pa.int32()))), pa.field('struct', pa.struct([middle, pa.field('inner2', pa.int64())])), pa.field('list-struct', pa.list_(pa.field('item', pa.struct([pa.field('inner1', pa.int64()), pa.field('inner2', pa.int64())])))), pa.field('basic2', pa.int64())]
    arrs = [[0], [[1, 2]], [{'middle': {'inner': 3}, 'inner2': 4}], [[{'inner1': 5, 'inner2': 6}, {'inner1': 7, 'inner2': 8}]], [9]]
    table = pa.table(arrs, schema=pa.schema(fields))
    path = str(tempdir / 'test.orc')
    orc.write_table(table, path)
    orc_file = orc.ORCFile(path)
    result1 = orc_file.read()
    assert result1.equals(table)
    result2 = orc_file.read(columns=['basic', 'basic2'])
    assert result2.equals(table.select(['basic', 'basic2']))
    result3 = orc_file.read(columns=['list', 'struct', 'basic2'])
    assert result3.equals(table.select(['list', 'struct', 'basic2']))
    result4 = orc_file.read(columns=['struct.middle.inner'])
    expected4 = pa.table({'struct': [{'middle': {'inner': 3}}]})
    assert result4.equals(expected4)
    result5 = orc_file.read(columns=['struct.inner2'])
    expected5 = pa.table({'struct': [{'inner2': 4}]})
    assert result5.equals(expected5)
    result6 = orc_file.read(columns=['list', 'struct.middle.inner', 'struct.inner2'])
    assert result6.equals(table.select(['list', 'struct']))
    result7 = orc_file.read(columns=['list-struct.inner1'])
    expected7 = pa.table({'list-struct': [[{'inner1': 5}, {'inner1': 7}]]})
    assert result7.equals(expected7)
    result2 = orc_file.read(columns=[0, 4])
    assert result2.equals(table.select(['basic', 'basic2']))
    result3 = orc_file.read(columns=[1, 2, 3])
    assert result3.equals(table.select(['list', 'struct', 'list-struct']))
    with pytest.raises(IOError):
        orc_file.read(columns=['wrong'])
    with pytest.raises(ValueError):
        orc_file.read(columns=[5])