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
def test_reader_convert_entry(tmpdir):
    converters = {'x1': lambda arg: 'yyy', 'x2': lambda arg: 'YYY'}
    j = journal.Reader(path=tmpdir.strpath, converters=converters)
    val = j._convert_entry({'x1': b'abc', 'y1': b'\x80\x80', 'x2': [b'abc', b'def'], 'y2': [b'\x80\x80', b'\x80\x81']})
    assert val == {'x1': 'yyy', 'y1': b'\x80\x80', 'x2': ['YYY', 'YYY'], 'y2': [b'\x80\x80', b'\x80\x81']}