import sys
import struct
import traceback
import threading
import logging
from paramiko.common import (
from paramiko.config import SSHConfig
def clamp_value(minimum, val, maximum):
    return max(minimum, min(val, maximum))