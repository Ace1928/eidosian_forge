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
def create_branch_and_repo(klass, base, force_new_repo=False, format=None) -> 'Branch':
    """Create a new ControlDir, Branch and Repository at the url 'base'.

        This will use the current default ControlDirFormat unless one is
        specified, and use whatever
        repository format that that uses via controldir.create_branch and
        create_repository. If a shared repository is available that is used
        preferentially.

        The created Branch object is returned.

        Args:
          base: The URL to create the branch at.
          force_new_repo: If True a new repository is always created.
          format: If supplied, the format of branch to create.  If not
            supplied, the default is used.
        """
    controldir = klass.create(base, format)
    controldir._find_or_create_repository(force_new_repo)
    return cast('Branch', controldir.create_branch())