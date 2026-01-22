import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def fontInfoWOFFMetadataLicenseValidator(value):
    """
    Version 3+.
    """
    dictPrototype = dict(url=(str, False), text=(list, False), id=(str, False))
    if not genericDictValidator(value, dictPrototype):
        return False
    if 'text' in value:
        for text in value['text']:
            if not fontInfoWOFFMetadataTextValue(text):
                return False
    return True