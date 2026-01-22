from __future__ import annotations
from io import BytesIO
from typing import Sequence
from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._deprecate import deprecate
def getiptcinfo(im):
    """
    Get IPTC information from TIFF, JPEG, or IPTC file.

    :param im: An image containing IPTC data.
    :returns: A dictionary containing IPTC information, or None if
        no IPTC information block was found.
    """
    from . import JpegImagePlugin, TiffImagePlugin
    data = None
    if isinstance(im, IptcImageFile):
        return im.info
    elif isinstance(im, JpegImagePlugin.JpegImageFile):
        photoshop = im.info.get('photoshop')
        if photoshop:
            data = photoshop.get(1028)
    elif isinstance(im, TiffImagePlugin.TiffImageFile):
        try:
            data = im.tag.tagdata[TiffImagePlugin.IPTC_NAA_CHUNK]
        except (AttributeError, KeyError):
            pass
    if data is None:
        return None

    class FakeImage:
        pass
    im = FakeImage()
    im.__class__ = IptcImageFile
    im.info = {}
    im.fp = BytesIO(data)
    try:
        im._open()
    except (IndexError, KeyError):
        pass
    return im.info