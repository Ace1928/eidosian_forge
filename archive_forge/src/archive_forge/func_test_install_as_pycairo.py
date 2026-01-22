import array
import base64
import contextlib
import gc
import io
import math
import os
import shutil
import sys
import tempfile
import cairocffi
import pikepdf
import pytest
from . import (
def test_install_as_pycairo():
    cairocffi.install_as_pycairo()
    import cairo
    assert cairo is cairocffi