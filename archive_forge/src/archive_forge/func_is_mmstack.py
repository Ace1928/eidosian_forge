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
def is_mmstack(self) -> bool:
    """File has Micro-Manager stack format."""
    meta = self.micromanager_metadata
    if meta is not None and 'Summary' in meta and ('IndexMap' in meta) and (meta.get('MajorVersion', 1) == 0):
        self.is_uniform = True
        return True
    return False