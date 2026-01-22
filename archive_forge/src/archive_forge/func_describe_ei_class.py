from .enums import (
from .constants import (
from ..common.utils import bytes2hex
def describe_ei_class(x):
    return _DESCR_EI_CLASS.get(x, _unknown)