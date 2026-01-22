from __future__ import annotations
import logging # isort:skip
import os
from os.path import join
from pathlib import Path
from typing import (
import yaml
from .util.deprecation import deprecated
from .util.paths import bokehjs_path, server_path
def secret_key_bytes(self) -> bytes | None:
    """ Return the secret_key, converted to bytes and cached.

        """
    if not hasattr(self, '_secret_key_bytes'):
        key = self.secret_key()
        if key is None:
            self._secret_key_bytes = None
        else:
            self._secret_key_bytes = key.encode('utf-8')
    return self._secret_key_bytes