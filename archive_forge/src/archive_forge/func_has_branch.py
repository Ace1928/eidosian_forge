from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
def has_branch(self, name=None):
    """Tell if this controldir contains a branch.

        Note: if you're going to open the branch, you should just go ahead
        and try, and not ask permission first.  (This method just opens the
        branch and discards it, and that's somewhat expensive.)
        """
    try:
        self.open_branch(name, ignore_fallbacks=True)
        return True
    except errors.NotBranchError:
        return False