import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def fontInfoWOFFMetadataExtensionsValidator(value):
    """
    Version 3+.
    """
    if not isinstance(value, list):
        return False
    if not value:
        return False
    for extension in value:
        if not fontInfoWOFFMetadataExtensionValidator(extension):
            return False
    return True