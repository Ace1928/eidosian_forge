from .enums import (
from .constants import (
from ..common.utils import bytes2hex
def describe_e_machine(x):
    return _DESCR_E_MACHINE.get(x, _unknown)