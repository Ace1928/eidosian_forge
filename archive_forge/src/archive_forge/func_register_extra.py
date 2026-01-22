from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
def register_extra(self, format):
    """Register a format that can not be used in a metadir.

        This is mainly useful to allow custom repository formats, such as older
        Bazaar formats and foreign formats, to be tested.
        """
    self._extra_formats.append(registry._ObjectGetter(format))