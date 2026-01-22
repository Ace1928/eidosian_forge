from .enums import (
from .constants import (
from ..common.utils import bytes2hex
def describe_dt_flags(x):
    return ' '.join((key[3:] for key, val in sorted(ENUM_DT_FLAGS.items(), key=lambda t: t[1]) if x & val))