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
@_keyGenerator('ecdsa')
def generateECDSAkey(options):
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.asymmetric import ec
    if not options['bits']:
        options['bits'] = 256
    curve = b'ecdsa-sha2-nistp' + str(options['bits']).encode('ascii')
    keyPrimitive = ec.generate_private_key(curve=keys._curveTable[curve], backend=default_backend())
    key = keys.Key(keyPrimitive)
    _saveKey(key, options)