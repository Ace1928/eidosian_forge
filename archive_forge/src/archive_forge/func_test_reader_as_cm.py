from __future__ import print_function
import contextlib
import datetime
import errno
import logging
import os
import time
import uuid
import sys
import traceback
from systemd import journal, id128
from systemd.journal import _make_line
import pytest
def test_reader_as_cm(tmpdir):
    j = journal.Reader(path=tmpdir.strpath)
    with j:
        assert not j.closed
    assert j.closed
    with pytest.raises(OSError):
        next(j)