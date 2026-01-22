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
def test_uuid_type_pickle(pickle_module):
    for proto in range(0, pickle_module.HIGHEST_PROTOCOL + 1):
        ty = UuidType()
        ser = pickle_module.dumps(ty, protocol=proto)
        del ty
        ty = pickle_module.loads(ser)
        wr = weakref.ref(ty)
        assert ty.extension_name == 'pyarrow.tests.UuidType'
        del ty
        assert wr() is None