import json
import os
import pyarrow as pa
import pyarrow.jvm as pa_jvm
import pytest
import sys
import xml.etree.ElementTree as ET
@pytest.mark.parametrize('pa_type,py_data,jvm_type,jvm_spec', [(pa.bool_(), [True, False, True, True], 'BitVector', '{"name":"bool"}'), (pa.uint8(), list(range(128)), 'UInt1Vector', '{"name":"int","bitWidth":8,"isSigned":false}'), (pa.uint16(), list(range(128)), 'UInt2Vector', '{"name":"int","bitWidth":16,"isSigned":false}'), (pa.uint32(), list(range(128)), 'UInt4Vector', '{"name":"int","bitWidth":32,"isSigned":false}'), (pa.uint64(), list(range(128)), 'UInt8Vector', '{"name":"int","bitWidth":64,"isSigned":false}'), (pa.int8(), list(range(128)), 'TinyIntVector', '{"name":"int","bitWidth":8,"isSigned":true}'), (pa.int16(), list(range(128)), 'SmallIntVector', '{"name":"int","bitWidth":16,"isSigned":true}'), (pa.int32(), list(range(128)), 'IntVector', '{"name":"int","bitWidth":32,"isSigned":true}'), (pa.int64(), list(range(128)), 'BigIntVector', '{"name":"int","bitWidth":64,"isSigned":true}'), (pa.float32(), list(range(128)), 'Float4Vector', '{"name":"floatingpoint","precision":"SINGLE"}'), (pa.float64(), list(range(128)), 'Float8Vector', '{"name":"floatingpoint","precision":"DOUBLE"}'), (pa.timestamp('s'), list(range(128)), 'TimeStampSecVector', '{"name":"timestamp","unit":"SECOND","timezone":null}'), (pa.timestamp('ms'), list(range(128)), 'TimeStampMilliVector', '{"name":"timestamp","unit":"MILLISECOND","timezone":null}'), (pa.timestamp('us'), list(range(128)), 'TimeStampMicroVector', '{"name":"timestamp","unit":"MICROSECOND","timezone":null}'), (pa.timestamp('ns'), list(range(128)), 'TimeStampNanoVector', '{"name":"timestamp","unit":"NANOSECOND","timezone":null}'), (pa.date32(), list(range(128)), 'DateDayVector', '{"name":"date","unit":"DAY"}'), (pa.date64(), list(range(128)), 'DateMilliVector', '{"name":"date","unit":"MILLISECOND"}')])
def test_jvm_record_batch(root_allocator, pa_type, py_data, jvm_type, jvm_spec):
    cls = 'org.apache.arrow.vector.{}'.format(jvm_type)
    jvm_vector = jpype.JClass(cls)('vector', root_allocator)
    jvm_vector.allocateNew(len(py_data))
    for i, val in enumerate(py_data):
        if jvm_type in {'UInt1Vector', 'UInt2Vector'}:
            val = jpype.JInt(val)
        jvm_vector.setSafe(i, val)
    jvm_vector.setValueCount(len(py_data))
    spec = {'name': 'field_name', 'nullable': False, 'type': json.loads(jvm_spec), 'children': []}
    jvm_field = _jvm_field(json.dumps(spec))
    jvm_fields = jpype.JClass('java.util.ArrayList')()
    jvm_fields.add(jvm_field)
    jvm_vectors = jpype.JClass('java.util.ArrayList')()
    jvm_vectors.add(jvm_vector)
    jvm_vsr = jpype.JClass('org.apache.arrow.vector.VectorSchemaRoot')
    jvm_vsr = jvm_vsr(jvm_fields, jvm_vectors, len(py_data))
    py_record_batch = pa.RecordBatch.from_arrays([pa.array(py_data, type=pa_type)], ['col'])
    jvm_record_batch = pa_jvm.record_batch(jvm_vsr)
    assert py_record_batch.equals(jvm_record_batch)