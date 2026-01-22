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
def test_ext_type_basics():
    ty = UuidType()
    assert ty.extension_name == 'pyarrow.tests.UuidType'