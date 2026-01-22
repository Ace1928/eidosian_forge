import pytest
import decimal
import datetime
import pyarrow as pa
from pyarrow import fs
from pyarrow.tests import util
@pytest.mark.snappy
def test_buffer_readwrite_with_writeoptions():
    from pyarrow import orc
    buffer_output_stream = pa.BufferOutputStream()
    a = pa.array([1, None, 3, None])
    b = pa.array([None, 'Arrow', None, 'ORC'])
    table = pa.table({'int64': a, 'utf8': b})
    orc.write_table(table, buffer_output_stream, compression='snappy', file_version='0.11', row_index_stride=5000, compression_block_size=32768)
    buffer_reader = pa.BufferReader(buffer_output_stream.getvalue())
    orc_file = orc.ORCFile(buffer_reader)
    output_table = orc_file.read()
    assert table.equals(output_table)
    assert orc_file.compression == 'SNAPPY'
    assert orc_file.file_version == '0.11'
    assert orc_file.row_index_stride == 5000
    assert orc_file.compression_size == 32768
    buffer_output_stream = pa.BufferOutputStream()
    with pytest.warns(FutureWarning):
        orc.write_table(buffer_output_stream, table, compression='uncompressed', file_version='0.11', row_index_stride=20000, compression_block_size=16384)
    buffer_reader = pa.BufferReader(buffer_output_stream.getvalue())
    orc_file = orc.ORCFile(buffer_reader)
    output_table = orc_file.read()
    assert table.equals(output_table)
    assert orc_file.compression == 'UNCOMPRESSED'
    assert orc_file.file_version == '0.11'
    assert orc_file.row_index_stride == 20000
    assert orc_file.compression_size == 16384