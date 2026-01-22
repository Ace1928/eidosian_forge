import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def fontInfoStyleMapStyleNameValidator(value):
    """
    Version 2+.
    """
    options = ['regular', 'italic', 'bold', 'bold italic']
    return value in options