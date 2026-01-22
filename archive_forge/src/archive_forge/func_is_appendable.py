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
@property
def is_appendable(self) -> bool:
    """Pages can be appended to file without corrupting."""
    return not (self.is_ome or self.is_lsm or self.is_stk or self.is_imagej or self.is_fluoview or self.is_micromanager)