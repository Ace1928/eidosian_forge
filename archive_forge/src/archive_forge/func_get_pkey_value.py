from typing import List, Optional, Tuple
from pyglet.libs.win32 import com
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import CLSCTX_INPROC_SERVER
from pyglet.libs.win32.types import *
from pyglet.media.devices import base
from pyglet.util import debug_print
@staticmethod
def get_pkey_value(store: IPropertyStore, pkey: PROPERTYKEY):
    try:
        propvar = PROPVARIANT()
        store.GetValue(pkey, byref(propvar))
        value = propvar.pwszVal
        ole32.PropVariantClear(byref(propvar))
    except Exception:
        value = 'Unknown'
    return value