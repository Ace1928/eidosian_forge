from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
@classmethod
def find_controldirs(klass, transport, evaluate=None, list_current=None):
    """Find control dirs recursively from current location.

        This is intended primarily as a building block for more sophisticated
        functionality, like finding trees under a directory, or finding
        branches that use a given repository.

        Args:
          evaluate: An optional callable that yields recurse, value,
            where recurse controls whether this controldir is recursed into
            and value is the value to yield.  By default, all bzrdirs
            are recursed into, and the return value is the controldir.
          list_current: if supplied, use this function to list the current
            directory, instead of Transport.list_dir

        Returns:
          a generator of found bzrdirs, or whatever evaluate returns.
        """
    if list_current is None:

        def list_current(transport):
            return transport.list_dir('')
    if evaluate is None:

        def evaluate(controldir):
            return (True, controldir)
    pending = [transport]
    while len(pending) > 0:
        current_transport = pending.pop()
        recurse = True
        try:
            controldir = klass.open_from_transport(current_transport)
        except (errors.NotBranchError, errors.PermissionDenied, errors.UnknownFormatError):
            pass
        else:
            recurse, value = evaluate(controldir)
            yield value
        try:
            subdirs = list_current(current_transport)
        except (_mod_transport.NoSuchFile, errors.PermissionDenied):
            continue
        if recurse:
            for subdir in sorted(subdirs, reverse=True):
                pending.append(current_transport.clone(subdir))