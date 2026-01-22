import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def fontInfoPostscriptWindowsCharacterSetValidator(value):
    """
    Version 2+.
    """
    validValues = list(range(1, 21))
    if value not in validValues:
        return False
    return True