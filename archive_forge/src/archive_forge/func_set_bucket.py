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
def set_bucket(self, bucket: Bucket) -> None:
    """Put the bucket into the cache."""
    self.dump_bytecode(bucket)