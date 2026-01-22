from .enums import (
from .constants import (
from ..common.utils import bytes2hex
def describe_sh_type(x):
    if x in _DESCR_SH_TYPE:
        return _DESCR_SH_TYPE.get(x)
    elif x >= ENUM_SH_TYPE_BASE['SHT_LOOS'] and x < ENUM_SH_TYPE_BASE['SHT_GNU_versym']:
        return 'loos+0x%lx' % (x - ENUM_SH_TYPE_BASE['SHT_LOOS'])
    else:
        return _unknown