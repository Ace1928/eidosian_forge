import binascii
import os
import mmap
import sys
import time
import errno
from io import BytesIO
from smmap import (
import hashlib
from gitdb.const import (
def to_hex_sha(sha):
    """:return: hexified version  of sha"""
    if len(sha) == 40:
        return sha
    return bin_to_hex(sha)