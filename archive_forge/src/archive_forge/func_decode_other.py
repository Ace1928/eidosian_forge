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
def decode_other(data: bytes | None, index: int, /, *, jpegtables: bytes | None=None, jpegheader: bytes | None=None, _fullsize: bool=False) -> tuple[NDArray[Any] | None, tuple[int, int, int, int, int], tuple[int, int, int, int]]:
    segmentindex, shape = indices(index)
    if data is None:
        if _fullsize:
            shape = pad_none(shape)
        return (data, segmentindex, shape)
    if self.fillorder == 2:
        data = imagecodecs.bitorder_decode(data)
    if decompress is not None:
        size = shape[0] * shape[1] * shape[2] * shape[3]
        data = decompress(data, out=size * dtype.itemsize)
    data_array = unpack(data)
    data_array = reshape(data_array, segmentindex, shape)
    data_array = data_array.astype('=' + dtype.char, copy=False)
    if unpredict is not None:
        data_array = unpredict(data_array, axis=-2, out=data_array)
    if _fullsize:
        data_array, shape = pad(data_array, shape)
    return (data_array, segmentindex, shape)