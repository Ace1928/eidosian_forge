from .enums import (
from .constants import (
from ..common.utils import bytes2hex
def describe_symbol_other(x):
    vis = describe_symbol_visibility(x['visibility'])
    if x['local'] > 1 and x['local'] < 7:
        return vis + ' ' + describe_symbol_local(x['local'])
    return vis