from collections.abc import Iterable
import datetime
import decimal
import hypothesis as h
import hypothesis.strategies as st
import itertools
import pytest
import struct
import subprocess
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
def test_union_from_sparse():
    binary = pa.array([b'a', b' ', b'b', b'c', b' ', b' ', b'd'], type='binary')
    int64 = pa.array([0, 1, 0, 0, 2, 3, 0], type='int64')
    types = pa.array([0, 1, 0, 0, 1, 1, 0], type='int8')
    logical_types = pa.array([11, 13, 11, 11, 13, 13, 11], type='int8')
    py_value = [b'a', 1, b'b', b'c', 2, 3, b'd']

    def check_result(result, expected_field_names, expected_type_codes, expected_type_code_values):
        result.validate(full=True)
        assert result.to_pylist() == py_value
        actual_field_names = [result.type[i].name for i in range(result.type.num_fields)]
        assert actual_field_names == expected_field_names
        assert result.type.mode == 'sparse'
        assert result.type.type_codes == expected_type_codes
        assert expected_type_code_values.equals(result.type_codes)
        assert result.field(0).equals(binary)
        assert result.field(1).equals(int64)
        with pytest.raises(pa.ArrowTypeError):
            result.offsets
        with pytest.raises(KeyError):
            result.field(-1)
        with pytest.raises(KeyError):
            result.field(2)
    check_result(pa.UnionArray.from_sparse(types, [binary, int64]), expected_field_names=['0', '1'], expected_type_codes=[0, 1], expected_type_code_values=types)
    check_result(pa.UnionArray.from_sparse(types, [binary, int64], ['bin', 'int']), expected_field_names=['bin', 'int'], expected_type_codes=[0, 1], expected_type_code_values=types)
    check_result(pa.UnionArray.from_sparse(logical_types, [binary, int64], type_codes=[11, 13]), expected_field_names=['0', '1'], expected_type_codes=[11, 13], expected_type_code_values=logical_types)
    check_result(pa.UnionArray.from_sparse(logical_types, [binary, int64], ['bin', 'int'], [11, 13]), expected_field_names=['bin', 'int'], expected_type_codes=[11, 13], expected_type_code_values=logical_types)
    arr = pa.UnionArray.from_sparse(logical_types, [binary, int64])
    with pytest.raises(pa.ArrowInvalid):
        arr.validate(full=True)
    arr = pa.UnionArray.from_sparse(types, [binary, int64], type_codes=[11, 13])
    with pytest.raises(pa.ArrowInvalid):
        arr.validate(full=True)
    with pytest.raises(pa.ArrowInvalid):
        arr = pa.UnionArray.from_sparse(logical_types, [binary, int64[1:]])