import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def isDictEnough(value):
    """
    Some objects will likely come in that aren't
    dicts but are dict-ish enough.
    """
    if isinstance(value, Mapping):
        return True
    for attr in ('keys', 'values', 'items'):
        if not hasattr(value, attr):
            return False
    return True