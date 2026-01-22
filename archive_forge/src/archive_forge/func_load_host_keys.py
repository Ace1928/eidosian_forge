import errno
import getpass
import logging
import os
import socket
import subprocess
import sys
from binascii import hexlify
from typing import Dict, Optional, Set, Tuple, Type
from .. import bedding, config, errors, osutils, trace, ui
import weakref
def load_host_keys():
    """
    Load system host keys (probably doesn't work on windows) and any
    "discovered" keys from previous sessions.
    """
    global SYSTEM_HOSTKEYS, BRZ_HOSTKEYS
    try:
        SYSTEM_HOSTKEYS = paramiko.util.load_host_keys(os.path.expanduser('~/.ssh/known_hosts'))
    except OSError as e:
        trace.mutter('failed to load system host keys: ' + str(e))
    brz_hostkey_path = _ssh_host_keys_config_dir()
    try:
        BRZ_HOSTKEYS = paramiko.util.load_host_keys(brz_hostkey_path)
    except OSError as e:
        trace.mutter('failed to load brz host keys: ' + str(e))
        save_host_keys()