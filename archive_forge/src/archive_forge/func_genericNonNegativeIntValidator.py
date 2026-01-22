import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def genericNonNegativeIntValidator(value):
    """
    Generic. (Added at version 3.)
    """
    if not isinstance(value, int):
        return False
    if value < 0:
        return False
    return True