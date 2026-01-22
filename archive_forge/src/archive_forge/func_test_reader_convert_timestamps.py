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
def test_reader_convert_timestamps(tmpdir):
    j = journal.Reader(path=tmpdir.strpath)
    val = j._convert_field('_SOURCE_REALTIME_TIMESTAMP', 1641651559324187)
    if sys.version_info >= (3,):
        assert val.tzinfo is not None
    val = j._convert_field('__REALTIME_TIMESTAMP', 1641651559324187)
    if sys.version_info >= (3,):
        assert val.tzinfo is not None
    val = j._convert_field('COREDUMP_TIMESTAMP', 1641651559324187)
    if sys.version_info >= (3,):
        assert val.tzinfo is not None