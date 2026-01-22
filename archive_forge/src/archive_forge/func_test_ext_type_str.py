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
def test_ext_type_str():
    ty = IntegerType()
    expected = 'extension<pyarrow.tests.IntegerType<IntegerType>>'
    assert str(ty) == expected
    assert pa.DataType.__str__(ty) == expected