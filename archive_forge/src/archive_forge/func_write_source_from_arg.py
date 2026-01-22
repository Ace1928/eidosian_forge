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
def write_source_from_arg(source, mode='wb'):
    """
    Retrieve a open stream for writing to the source provided.

    The result stream will be open by a handler that would write raw bytes and
    transparently take care of the compression, remote authentication,
    network transfer, format encoding, and data writing.

    .. versionadded:: 1.4.0
    """
    if source is None:
        return StdoutSource()
    return _resolve_source_from_arg(source, _WRITERS)