from __future__ import absolute_import, print_function, division
import os
import io
import gzip
import sys
import bz2
import zipfile
from contextlib import contextmanager
import subprocess
import logging
from petl.errors import ArgumentError
from petl.compat import urlopen, StringIO, BytesIO, string_types, PY2
def register_reader(protocol, handler_class):
    """
    Register handler for automatic reading using a remote protocol.

    Use of the handler is determined matching the `protocol` with the scheme
    part of the url in ``from...()`` function (e.g: `http://`).

    .. versionadded:: 1.5.0
    """
    _register_handler(protocol, handler_class, _READERS)