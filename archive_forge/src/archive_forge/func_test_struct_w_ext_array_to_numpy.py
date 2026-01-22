import contextlib
import os
import shutil
import subprocess
import weakref
from uuid import uuid4, UUID
import sys
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
def test_struct_w_ext_array_to_numpy(struct_w_ext_data):
    result = struct_w_ext_data[0].to_numpy(zero_copy_only=False)
    expected = np.array([{'f0': 1}, {'f0': 2}, {'f0': 3}], dtype=object)
    np.testing.assert_array_equal(result, expected)
    result = struct_w_ext_data[1].to_numpy(zero_copy_only=False)
    expected = np.array([{'f1': b'123'}, {'f1': b'456'}, {'f1': b'789'}], dtype=object)
    np.testing.assert_array_equal(result, expected)