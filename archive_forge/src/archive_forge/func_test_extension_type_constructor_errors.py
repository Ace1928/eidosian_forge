import os
import subprocess
import sys
import pytest
import pyarrow as pa
from pyarrow.lib import ArrowInvalid
@pytest.mark.parametrize('klass', [pa.Field, pa.Schema, pa.ChunkedArray, pa.RecordBatch, pa.Table, pa.Buffer, pa.Array, pa.Tensor, pa.DataType, pa.ListType, pa.LargeListType, pa.FixedSizeListType, pa.UnionType, pa.SparseUnionType, pa.DenseUnionType, pa.StructType, pa.Time32Type, pa.Time64Type, pa.TimestampType, pa.Decimal128Type, pa.Decimal256Type, pa.DictionaryType, pa.FixedSizeBinaryType, pa.NullArray, pa.NumericArray, pa.IntegerArray, pa.FloatingPointArray, pa.BooleanArray, pa.Int8Array, pa.Int16Array, pa.Int32Array, pa.Int64Array, pa.UInt8Array, pa.UInt16Array, pa.UInt32Array, pa.UInt64Array, pa.ListArray, pa.LargeListArray, pa.MapArray, pa.FixedSizeListArray, pa.UnionArray, pa.BinaryArray, pa.StringArray, pa.FixedSizeBinaryArray, pa.DictionaryArray, pa.Date32Array, pa.Date64Array, pa.TimestampArray, pa.Time32Array, pa.Time64Array, pa.DurationArray, pa.Decimal128Array, pa.Decimal256Array, pa.StructArray, pa.RunEndEncodedArray, pa.Scalar, pa.BooleanScalar, pa.Int8Scalar, pa.Int16Scalar, pa.Int32Scalar, pa.Int64Scalar, pa.UInt8Scalar, pa.UInt16Scalar, pa.UInt32Scalar, pa.UInt64Scalar, pa.HalfFloatScalar, pa.FloatScalar, pa.DoubleScalar, pa.Decimal128Scalar, pa.Decimal256Scalar, pa.Date32Scalar, pa.Date64Scalar, pa.Time32Scalar, pa.Time64Scalar, pa.TimestampScalar, pa.DurationScalar, pa.StringScalar, pa.BinaryScalar, pa.FixedSizeBinaryScalar, pa.ListScalar, pa.LargeListScalar, pa.MapScalar, pa.FixedSizeListScalar, pa.UnionScalar, pa.StructScalar, pa.DictionaryScalar, pa.RunEndEncodedScalar, pa.ipc.Message, pa.ipc.MessageReader, pa.MemoryPool, pa.LoggingMemoryPool, pa.ProxyMemoryPool])
def test_extension_type_constructor_errors(klass):
    msg = "Do not call {cls}'s constructor directly, use .* instead."
    with pytest.raises(TypeError, match=msg.format(cls=klass.__name__)):
        klass()