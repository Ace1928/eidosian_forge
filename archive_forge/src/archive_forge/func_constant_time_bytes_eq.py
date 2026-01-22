import sys
import struct
import traceback
import threading
import logging
from paramiko.common import (
from paramiko.config import SSHConfig
def constant_time_bytes_eq(a, b):
    if len(a) != len(b):
        return False
    res = 0
    for i in range(len(a)):
        res |= byte_ord(a[i]) ^ byte_ord(b[i])
    return res == 0