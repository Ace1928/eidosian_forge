from typing import Dict, List, Tuple
from . import errors, registry
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext
def install_lazy_named_hook(hookpoints_module, hookpoints_name, hook_name, a_callable, name):
    """Install a callable in to a hook lazily, and label it name.

    :param hookpoints_module: Module name of the hook points.
    :param hookpoints_name: Name of the hook points.
    :param hook_name: A hook name.
    :param callable: a callable to call for the hook.
    :param name: A name to associate a_callable with, to show users what is
        running.
    """
    key = (hookpoints_module, hookpoints_name, hook_name)
    obj_getter = registry._ObjectGetter(a_callable)
    _lazy_hooks.setdefault(key, []).append((obj_getter, name))