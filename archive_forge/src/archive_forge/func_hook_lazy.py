from typing import Dict, List, Tuple
from . import errors, registry
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext
def hook_lazy(self, callback_module, callback_member, callback_label):
    """Lazily register a callback to be called when this HookPoint fires.

        :param callback_module: Module of the callable to use when this
            HookPoint fires.
        :param callback_member: Member name of the callback.
        :param callback_label: A label to show in the UI while this callback is
            processing.
        """
    obj_getter = registry._LazyObjectGetter(callback_module, callback_member)
    self._callbacks.append((obj_getter, callback_label))