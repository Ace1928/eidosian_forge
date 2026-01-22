import os
import sqlite3
from io import BytesIO
from os.path import dirname
from os.path import join as pjoin
from ..testing import suppress_warnings
import unittest
import pytest
from .. import nifti1
from ..optpkg import optional_package
def test_cursor_conflict(self):
    rwc = self._db.readwrite_cursor
    statement = ('INSERT INTO directory (path, mtime) VALUES (?, ?)', ('/tmp', 0))
    with pytest.raises(sqlite3.IntegrityError):
        with rwc() as c1, rwc() as c2:
            c1.execute(*statement)
            c2.execute(*statement)