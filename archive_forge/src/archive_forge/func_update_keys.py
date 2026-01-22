import binascii
import importlib.util
import io
import itertools
import os
import posixpath
import shutil
import stat
import struct
import sys
import threading
import time
import contextlib
import pathlib
def update_keys(c):
    nonlocal key0, key1, key2
    key0 = crc32(c, key0)
    key1 = key1 + (key0 & 255) & 4294967295
    key1 = key1 * 134775813 + 1 & 4294967295
    key2 = crc32(key1 >> 24, key2)