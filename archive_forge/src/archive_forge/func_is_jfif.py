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
@cached_property
def is_jfif(self) -> bool:
    """JPEG compressed segments contain JFIF metadata."""
    if self.compression not in {6, 7, 34892, 33007} or len(self.dataoffsets) < 1 or self.dataoffsets[0] == 0 or (len(self.databytecounts) < 1) or (self.databytecounts[0] < 11):
        return False
    fh = self.parent.filehandle
    fh.seek(self.dataoffsets[0] + 6)
    data = fh.read(4)
    return data == b'JFIF'