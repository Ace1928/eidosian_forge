import sys
import warnings
from collections import UserList
import gi
from gi.repository import GObject
def enable_poppler():
    if _check_enabled('poppler'):
        return
    gi.require_version('Poppler', '0.18')
    from gi.repository import Poppler
    _patch_module('poppler', Poppler)
    _patch(Poppler, 'pypoppler_version', (1, 0, 0))