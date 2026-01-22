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
@pytest.mark.parametrize('data,type_factory', (([[1, 2, 3]], lambda: pa.list_(IntegerType())), ([{'foo': 1}], lambda: pa.struct([('foo', IntegerType())])), ([[{'foo': 1}]], lambda: pa.list_(pa.struct([('foo', IntegerType())]))), ([{'foo': [1, 2, 3]}], lambda: pa.struct([('foo', pa.list_(IntegerType()))]))))
def test_cast_nested_extension_types(data, type_factory):
    ty = type_factory()
    a = pa.array(data)
    b = a.cast(ty)
    assert b.type == ty
    assert b.cast(a.type)