import sys
import struct
import traceback
import threading
import logging
from paramiko.common import (
from paramiko.config import SSHConfig
def lookup_ssh_host_config(hostname, config):
    """
    Provided only as a backward-compatible wrapper around `.SSHConfig`.
    """
    return config.lookup(hostname)