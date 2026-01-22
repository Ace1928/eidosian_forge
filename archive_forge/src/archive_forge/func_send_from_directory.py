from __future__ import annotations
import io
import mimetypes
import os
import pkgutil
import re
import sys
import typing as t
import unicodedata
from datetime import datetime
from time import time
from urllib.parse import quote
from zlib import adler32
from markupsafe import escape
from ._internal import _DictAccessorProperty
from ._internal import _missing
from ._internal import _TAccessorValue
from .datastructures import Headers
from .exceptions import NotFound
from .exceptions import RequestedRangeNotSatisfiable
from .security import safe_join
from .wsgi import wrap_file
def send_from_directory(directory: os.PathLike | str, path: os.PathLike | str, environ: WSGIEnvironment, **kwargs: t.Any) -> Response:
    """Send a file from within a directory using :func:`send_file`.

    This is a secure way to serve files from a folder, such as static
    files or uploads. Uses :func:`~werkzeug.security.safe_join` to
    ensure the path coming from the client is not maliciously crafted to
    point outside the specified directory.

    If the final path does not point to an existing regular file,
    returns a 404 :exc:`~werkzeug.exceptions.NotFound` error.

    :param directory: The directory that ``path`` must be located under. This *must not*
        be a value provided by the client, otherwise it becomes insecure.
    :param path: The path to the file to send, relative to ``directory``. This is the
        part of the path provided by the client, which is checked for security.
    :param environ: The WSGI environ for the current request.
    :param kwargs: Arguments to pass to :func:`send_file`.

    .. versionadded:: 2.0
        Adapted from Flask's implementation.
    """
    path = safe_join(os.fspath(directory), os.fspath(path))
    if path is None:
        raise NotFound()
    if '_root_path' in kwargs:
        path = os.path.join(kwargs['_root_path'], path)
    if not os.path.isfile(path):
        raise NotFound()
    return send_file(path, environ, **kwargs)