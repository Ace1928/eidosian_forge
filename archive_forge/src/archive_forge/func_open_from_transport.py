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
def open_from_transport(klass, transport: _mod_transport.Transport, _unsupported=False, probers=None) -> 'ControlDir':
    """Open a controldir within a particular directory.

        Args:
          transport: Transport containing the controldir.
          _unsupported: private.
        """
    for hook in klass.hooks['pre_open']:
        hook(transport)
    base = transport.base

    def find_format(transport):
        return (transport, ControlDirFormat.find_format(transport, probers=probers))

    def redirected(transport, e, redirection_notice):
        redirected_transport = transport._redirected_to(e.source, e.target)
        if redirected_transport is None:
            raise errors.NotBranchError(base)
        trace.note(gettext('{0} is{1} redirected to {2}').format(transport.base, e.permanently, redirected_transport.base))
        return redirected_transport
    try:
        transport, format = _mod_transport.do_catching_redirections(find_format, transport, redirected)
    except errors.TooManyRedirections:
        raise errors.NotBranchError(base)
    format.check_support_status(_unsupported)
    return cast('ControlDir', format.open(transport, _found=True))