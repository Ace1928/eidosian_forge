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
def test_tensor_type_equality():
    tensor_type = pa.fixed_shape_tensor(pa.int8(), [2, 2, 3])
    assert tensor_type.extension_name == 'arrow.fixed_shape_tensor'
    tensor_type2 = pa.fixed_shape_tensor(pa.int8(), [2, 2, 3])
    tensor_type3 = pa.fixed_shape_tensor(pa.uint8(), [2, 2, 3])
    assert tensor_type == tensor_type2
    assert not tensor_type == tensor_type3