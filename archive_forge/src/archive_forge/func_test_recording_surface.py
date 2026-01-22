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
def test_recording_surface():
    text_pixels, recorded_pixels = _recording_surface_common((0, 0, 140, 80))
    assert recorded_pixels == text_pixels