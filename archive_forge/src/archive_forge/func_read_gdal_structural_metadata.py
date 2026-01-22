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
def read_gdal_structural_metadata(fh: FileHandle | BinaryIO, /) -> dict[str, str] | None:
    """Read non-TIFF GDAL structural metadata from file.

    Return None if the file does not contain valid GDAL structural metadata.
    The metadata can be used to optimize reading image data from a COG file.

    """
    fh.seek(0)
    try:
        if fh.read(2) not in {b'II', b'MM'}:
            raise ValueError('not a TIFF file')
        fh.seek({b'*': 8, b'+': 16}[fh.read(1)])
        header = fh.read(43).decode()
        if header[:30] != 'GDAL_STRUCTURAL_METADATA_SIZE=':
            return None
        size = int(header[30:36])
        lines = fh.read(size).decode()
    except Exception:
        return None
    result: dict[str, Any] = {}
    try:
        for line in lines.splitlines():
            if '=' in line:
                key, value = line.split('=', 1)
                result[key.strip()] = value.strip()
    except Exception as exc:
        logger().warning(f'<tifffile.read_gdal_structural_metadata> raised {exc!r}')
        return None
    return result