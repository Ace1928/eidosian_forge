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
def open_containing_tree_or_branch(klass, location, possible_transports=None):
    """Return the branch and working tree contained by a location.

        Returns (tree, branch, relpath).
        If there is no tree at containing the location, tree will be None.
        If there is no branch containing the location, an exception will be
        raised
        relpath is the portion of the path that is contained by the branch.
        """
    controldir, relpath = klass.open_containing(location, possible_transports=possible_transports)
    tree, branch = controldir._get_tree_branch()
    return (tree, branch, relpath)