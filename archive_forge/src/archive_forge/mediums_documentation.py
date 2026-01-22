import lazyops.libs.sqlcache.base
import os
import io
import zlib
import gzip
import codecs
import struct
import sqlite3
import errno
import pickletools
import dill as pkl
import os.path as op
import functools as ft
import contextlib as cl
from fileio.lib.types import File, FileLike
from typing import Any, Iterable, AsyncIterable, Optional
from lazyops.utils.pooler import ThreadPooler
from lazyops.libs.sqlcache.types import BaseMedium
from lazyops.libs.sqlcache.constants import (

        Deserialize bytes to value
        