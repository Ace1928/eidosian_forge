import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def pngValidator(path=None, data=None, fileObj=None):
    """
    Version 3+.

    This checks the signature of the image data.
    """
    assert path is not None or data is not None or fileObj is not None
    if path is not None:
        with open(path, 'rb') as f:
            signature = f.read(8)
    elif data is not None:
        signature = data[:8]
    elif fileObj is not None:
        pos = fileObj.tell()
        signature = fileObj.read(8)
        fileObj.seek(pos)
    if signature != pngSignature:
        return (False, 'Image does not begin with the PNG signature.')
    return (True, None)