from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
def register_extra_lazy(self, module_name, member_name):
    """Register a format lazily.
        """
    self._extra_formats.append(registry._LazyObjectGetter(module_name, member_name))