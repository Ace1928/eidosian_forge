from gitdb.test.lib import (
from gitdb.stream import DeltaApplyReader
from gitdb.pack import (
from gitdb.base import (
from gitdb.fun import delta_types
from gitdb.exc import UnsupportedOperation
from gitdb.util import to_bin_sha
import pytest
import os
import tempfile
def bin_sha_from_filename(filename):
    return to_bin_sha(os.path.splitext(os.path.basename(filename))[0][5:])