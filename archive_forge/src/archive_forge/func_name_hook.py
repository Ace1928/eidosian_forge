from typing import Dict, List, Tuple
from . import errors, registry
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext
def name_hook(self, a_callable, name):
    """Associate name with a_callable to show users what is running."""
    self._callable_names[a_callable] = name