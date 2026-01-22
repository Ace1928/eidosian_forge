import io
import re
import functools
import inspect
import os
import sys
import numbers
import warnings
from pathlib import Path, PurePath
from typing import (
from ase.atoms import Atoms
from importlib import import_module
from ase.parallel import parallel_function, parallel_generator
def filetype(filename: NameOrFile, read: bool=True, guess: bool=True) -> str:
    """Try to guess the type of the file.

    First, special signatures in the filename will be checked for.  If that
    does not identify the file type, then the first 2000 bytes of the file
    will be read and analysed.  Turn off this second part by using
    read=False.

    Can be used from the command-line also::

        $ ase info filename ...
    """
    orig_filename = filename
    if hasattr(filename, 'name'):
        filename = filename.name
    ext = None
    if isinstance(filename, str):
        if os.path.isdir(filename):
            if os.path.basename(os.path.normpath(filename)) == 'states':
                return 'eon'
            return 'bundletrajectory'
        if filename.startswith('postgres'):
            return 'postgresql'
        if filename.startswith('mysql') or filename.startswith('mariadb'):
            return 'mysql'
        root, compression = get_compression(filename)
        basename = os.path.basename(root)
        if '.' in basename:
            ext = os.path.splitext(basename)[1].strip('.').lower()
        for fmt in ioformats.values():
            if fmt.match_name(basename):
                return fmt.name
        if not read:
            if ext is None:
                raise UnknownFileTypeError('Could not guess file type')
            ioformat = extension2format.get(ext)
            if ioformat:
                return ioformat.name
            return ext
        if orig_filename == filename:
            fd = open_with_compression(filename, 'rb')
        else:
            fd = orig_filename
    else:
        fd = filename
        if fd is sys.stdin:
            return 'json'
    data = fd.read(PEEK_BYTES)
    if fd is not filename:
        fd.close()
    else:
        fd.seek(0)
    if len(data) == 0:
        raise UnknownFileTypeError('Empty file: ' + filename)
    try:
        return match_magic(data).name
    except UnknownFileTypeError:
        pass
    format = None
    if ext in extension2format:
        format = extension2format[ext].name
    if format is None and guess:
        format = ext
    if format is None:
        lines = data.splitlines()
        if lines and lines[0].strip().isdigit():
            return extension2format['xyz'].name
        raise UnknownFileTypeError('Could not guess file type')
    assert isinstance(format, str)
    return format