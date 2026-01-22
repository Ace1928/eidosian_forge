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
def test_journal_stream():
    with skip_oserror(errno.ENOENT):
        stream = journal.stream('test_journal.py')
    res = stream.write('message...\n')
    assert res in (11, None)
    print('printed message...', file=stream)