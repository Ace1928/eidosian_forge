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
def test_ext_scalar_from_storage():
    ty = UuidType()
    s = pa.ExtensionScalar.from_storage(ty, None)
    assert isinstance(s, pa.ExtensionScalar)
    assert s.type == ty
    assert s.is_valid is False
    assert s.value is None
    s = pa.ExtensionScalar.from_storage(ty, b'0123456789abcdef')
    assert isinstance(s, pa.ExtensionScalar)
    assert s.type == ty
    assert s.is_valid is True
    assert s.value == pa.scalar(b'0123456789abcdef', ty.storage_type)
    s = pa.ExtensionScalar.from_storage(ty, pa.scalar(None, ty.storage_type))
    assert isinstance(s, pa.ExtensionScalar)
    assert s.type == ty
    assert s.is_valid is False
    assert s.value is None
    s = pa.ExtensionScalar.from_storage(ty, pa.scalar(b'0123456789abcdef', ty.storage_type))
    assert isinstance(s, pa.ExtensionScalar)
    assert s.type == ty
    assert s.is_valid is True
    assert s.value == pa.scalar(b'0123456789abcdef', ty.storage_type)