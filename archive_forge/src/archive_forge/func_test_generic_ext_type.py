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
def test_generic_ext_type():
    period_type = PeriodType('D')
    assert period_type.extension_name == 'test.period'
    assert period_type.storage_type == pa.int64()
    assert period_type.__arrow_ext_class__() == pa.ExtensionArray