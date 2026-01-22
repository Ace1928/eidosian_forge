import pytest
import decimal
import datetime
import pyarrow as pa
from pyarrow import fs
from pyarrow.tests import util
def test_buffer_readwrite_with_bad_writeoptions():
    from pyarrow import orc
    buffer_output_stream = pa.BufferOutputStream()
    a = pa.array([1, None, 3, None])
    table = pa.table({'int64': a})
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, batch_size=0)
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, batch_size=-100)
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, batch_size=1024.23)
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, file_version=0.13)
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, file_version='1.1')
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, stripe_size=0)
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, stripe_size=-400)
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, stripe_size=4096.73)
    with pytest.raises(TypeError):
        orc.write_table(table, buffer_output_stream, compression=0)
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, compression='none')
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, compression='zlid')
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, compression_block_size=0)
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, compression_block_size=-200)
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, compression_block_size=1096.73)
    with pytest.raises(TypeError):
        orc.write_table(table, buffer_output_stream, compression_strategy=0)
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, compression_strategy='no')
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, compression_strategy='large')
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, row_index_stride=0)
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, row_index_stride=-800)
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, row_index_stride=3096.29)
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, padding_tolerance='cat')
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, dictionary_key_size_threshold='arrow')
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, dictionary_key_size_threshold=1.2)
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, dictionary_key_size_threshold=-3.2)
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, bloom_filter_columns='string')
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, bloom_filter_columns=[0, 1.4])
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, bloom_filter_columns={0, 2, -1})
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, bloom_filter_fpp='arrow')
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, bloom_filter_fpp=1.1)
    with pytest.raises(ValueError):
        orc.write_table(table, buffer_output_stream, bloom_filter_fpp=-0.1)