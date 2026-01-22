import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def fontInfoOpenTypeNameRecordsValidator(value):
    """
    Version 3+.
    """
    if not isinstance(value, list):
        return False
    dictPrototype = dict(nameID=(int, True), platformID=(int, True), encodingID=(int, True), languageID=(int, True), string=(str, True))
    for nameRecord in value:
        if not genericDictValidator(nameRecord, dictPrototype):
            return False
    return True