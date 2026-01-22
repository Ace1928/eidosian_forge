import contextlib
import errno
import re
import sys
import typing
from abc import ABC
from collections import OrderedDict
from collections.abc import MutableMapping
from types import TracebackType
from typing import Dict, Set, Optional, Union, Iterator, IO, Iterable, TYPE_CHECKING, Type
@classmethod
def load_from_path(cls, substvars_path, missing_ok=False):
    """Shorthand for initializing a Substvars from a file

        The return substvars will have `substvars_path` set to the provided path enabling
        `save()` to work out of the box. This also makes it easy to combine this with the
        context manager interface to automatically save the file again.

        >>> import os
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as tmpdir:
        ...    filename = os.path.join(tmpdir, "foo.substvars")
        ...    # Obviously, this does not exist
        ...    print("Exists before: " + str(os.path.exists(filename)))
        ...    with Substvars.load_from_path(filename, missing_ok=True) as svars:
        ...        svars.add_dependency("misc:Depends", "bar (>= 1.0)")
        ...    print("Exists after: " + str(os.path.exists(filename)))
        Exists before: False
        Exists after: True

        :param substvars_path: The path to load from
        :param missing_ok: If True, then the path does not have to exist (i.e.
          FileNotFoundError causes an empty Substvars object to be returned).  Combined
          with the context manager, this is useful for packaging helpers that want to
          append / update to the existing if it exists or create it if it does not exist.
        """
    substvars = cls()
    try:
        with open(substvars_path, 'r', encoding='utf-8') as fd:
            substvars.read_substvars(fd)
    except OSError as e:
        if e.errno != errno.ENOENT or not missing_ok:
            raise
    substvars.substvars_path = substvars_path
    return substvars