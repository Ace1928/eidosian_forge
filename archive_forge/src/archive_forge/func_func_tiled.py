from __future__ import annotations
import binascii
import collections
import datetime
import enum
import glob
import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import numpy
from typing import TYPE_CHECKING, BinaryIO, cast, final, overload
def func_tiled(page: TiffPage | TiffFrame | None, index: tuple[int | slice, ...], out: Any=out, filecache: FileCache=filecache, kwargs: dict[str, Any]=kwargs, /) -> None:
    if page is not None:
        filecache.open(page.parent.filehandle)
        out[index] = page.asarray(lock=lock, **kwargs)
        filecache.close(page.parent.filehandle)