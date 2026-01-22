import errno
import fnmatch
import marshal
import os
import pickle
import stat
import sys
import tempfile
import typing as t
from hashlib import sha1
from io import BytesIO
from types import CodeType
def bytecode_from_string(self, string: bytes) -> None:
    """Load bytecode from bytes."""
    self.load_bytecode(BytesIO(string))