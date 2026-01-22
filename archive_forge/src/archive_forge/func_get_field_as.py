import inspect
from functools import partial
from ..utils.module_loading import import_string
from .mountedtype import MountedType
from .unmountedtype import UnmountedType
def get_field_as(value, _as=None):
    """
    Get type mounted
    """
    if isinstance(value, MountedType):
        return value
    elif isinstance(value, UnmountedType):
        if _as is None:
            return value
        return _as.mounted(value)