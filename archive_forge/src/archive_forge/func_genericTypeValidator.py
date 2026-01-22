import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def genericTypeValidator(value, typ):
    """
    Generic. (Added at version 2.)
    """
    return isinstance(value, typ)