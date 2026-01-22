from __future__ import division, print_function
import sys
import os
import io
import re
import glob
import math
import zlib
import time
import json
import enum
import struct
import pathlib
import warnings
import binascii
import tempfile
import datetime
import threading
import collections
import multiprocessing
import concurrent.futures
import numpy
def write_empty(self, size):
    """Append size bytes to file. Position must be at end of file."""
    if size < 1:
        return
    self._fh.seek(size - 1, 1)
    self._fh.write(b'\x00')