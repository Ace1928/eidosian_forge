import os
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
import pyarrow.tests.util as test_util
def check_cython_example_module(mod):
    arr = pa.array([1, 2, 3])
    assert mod.get_array_length(arr) == 3
    with pytest.raises(TypeError, match='not an array'):
        mod.get_array_length(None)
    scal = pa.scalar(123)
    cast_scal = mod.cast_scalar(scal, pa.utf8())
    assert cast_scal == pa.scalar('123')
    with pytest.raises(NotImplementedError, match='casting scalars of type int64 to type list'):
        mod.cast_scalar(scal, pa.list_(pa.int64()))