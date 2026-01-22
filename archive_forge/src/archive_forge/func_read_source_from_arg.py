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
def read_source_from_arg(source):
    """
    Retrieve a open stream for reading from the source provided.

    The result stream will be open by a handler that would return raw bytes and
    transparently take care of the decompression, remote authentication,
    network transfer, format decoding, and data extraction.

    .. versionadded:: 1.4.0
    """
    if source is None:
        return StdinSource()
    return _resolve_source_from_arg(source, _READERS)