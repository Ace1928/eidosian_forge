from typing import Dict, List, Tuple
from . import errors, registry
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext
def name_hook_lazy(self, callable_module, callable_member, callable_name):
    self._lazy_callable_names[callable_module, callable_member] = callable_name