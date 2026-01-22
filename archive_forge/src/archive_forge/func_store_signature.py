from __future__ import annotations
import hashlib
import os
import sys
import typing as t
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
from hmac import HMAC
from pathlib import Path
from base64 import encodebytes
from jupyter_core.application import JupyterApp, base_flags
from traitlets import Any, Bool, Bytes, Callable, Enum, Instance, Integer, Unicode, default, observe
from traitlets.config import LoggingConfigurable, MultipleInstanceError
from . import NO_CONVERT, __version__, read, reads
def store_signature(self, digest, algorithm):
    """Store a signature in the db."""
    if self.db is None:
        return
    if not self.check_signature(digest, algorithm):
        self.db.execute('\n                INSERT INTO nbsignatures (algorithm, signature, last_seen)\n                VALUES (?, ?, ?)\n                ', (algorithm, digest, datetime.now(tz=timezone.utc)))
    else:
        self.db.execute('UPDATE nbsignatures SET last_seen = ? WHERE\n                algorithm = ? AND\n                signature = ?;\n                ', (datetime.now(tz=timezone.utc), algorithm, digest))
    self.db.commit()
    n, = self.db.execute('SELECT Count(*) FROM nbsignatures').fetchone()
    if n > self.cache_size:
        self.cull_db()