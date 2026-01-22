from .enums import (
from .constants import (
from ..common.utils import bytes2hex
def describe_p_type(x):
    if x in _DESCR_P_TYPE:
        return _DESCR_P_TYPE.get(x)
    elif x >= ENUM_P_TYPE_BASE['PT_LOOS'] and x <= ENUM_P_TYPE_BASE['PT_HIOS']:
        return 'LOOS+%lx' % (x - ENUM_P_TYPE_BASE['PT_LOOS'])
    else:
        return _unknown