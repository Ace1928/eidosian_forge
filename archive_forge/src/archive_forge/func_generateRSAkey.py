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
@_keyGenerator('rsa')
def generateRSAkey(options):
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.asymmetric import rsa
    if not options['bits']:
        options['bits'] = 2048
    keyPrimitive = rsa.generate_private_key(key_size=int(options['bits']), public_exponent=65537, backend=default_backend())
    key = keys.Key(keyPrimitive)
    _saveKey(key, options)