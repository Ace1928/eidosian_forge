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
def test_reader_converters(tmpdir):
    converters = {'xxx': lambda arg: 'yyy'}
    j = journal.Reader(path=tmpdir.strpath, converters=converters)
    val = j._convert_field('xxx', b'abc')
    assert val == 'yyy'
    val = j._convert_field('zzz', b'\x80\x80')
    assert val == b'\x80\x80'