import sys
import struct
import traceback
import threading
import logging
from paramiko.common import (
from paramiko.config import SSHConfig
def parse_ssh_config(file_obj):
    """
    Provided only as a backward-compatible wrapper around `.SSHConfig`.

    .. deprecated:: 2.7
        Use `SSHConfig.from_file` instead.
    """
    config = SSHConfig()
    config.parse(file_obj)
    return config