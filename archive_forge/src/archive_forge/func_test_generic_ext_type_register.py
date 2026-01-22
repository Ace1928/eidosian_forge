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
def test_generic_ext_type_register(registered_period_type):
    with pytest.raises(TypeError):
        pa.register_extension_type(pa.string())
    period_type = PeriodType('D')
    with pytest.raises(KeyError):
        pa.register_extension_type(period_type)