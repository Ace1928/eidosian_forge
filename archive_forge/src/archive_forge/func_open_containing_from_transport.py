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
def open_containing_from_transport(klass, a_transport, probers=None):
    """Open an existing branch which contains a_transport.base.

        This probes for a branch at a_transport, and searches upwards from there.

        Basically we keep looking up until we find the control directory or
        run into the root.  If there isn't one, raises NotBranchError.
        If there is one and it is either an unrecognised format or an unsupported
        format, UnknownFormatError or UnsupportedFormatError are raised.
        If there is one, it is returned, along with the unused portion of url.

        Returns: The ControlDir that contains the path, and a Unicode path
                for the rest of the URL.
        """
    url = a_transport.base
    while True:
        try:
            result = klass.open_from_transport(a_transport, probers=probers)
            return (result, urlutils.unescape(a_transport.relpath(url)))
        except errors.NotBranchError:
            pass
        except errors.PermissionDenied:
            pass
        try:
            new_t = a_transport.clone('..')
        except urlutils.InvalidURLJoin:
            raise errors.NotBranchError(path=url)
        if new_t.base == a_transport.base:
            raise errors.NotBranchError(path=url)
        a_transport = new_t