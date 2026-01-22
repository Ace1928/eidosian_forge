from __future__ import annotations
import getpass
import os
import platform
import socket
import sys
from collections.abc import Callable
from functools import wraps
from importlib import reload
from typing import Any, Dict, Optional
from twisted.conch.ssh import keys
from twisted.python import failure, filepath, log, usage
def assignkeygenerator(keygenerator):

    @wraps(keygenerator)
    def wrapper(*args, **kwargs):
        return keygenerator(*args, **kwargs)
    supportedKeyTypes[keyType] = wrapper
    return wrapper