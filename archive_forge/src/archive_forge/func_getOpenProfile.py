from __future__ import annotations
import sys
from enum import IntEnum
from . import Image
def getOpenProfile(profileFilename):
    """
    (pyCMS) Opens an ICC profile file.

    The PyCMSProfile object can be passed back into pyCMS for use in creating
    transforms and such (as in ImageCms.buildTransformFromOpenProfiles()).

    If ``profileFilename`` is not a valid filename for an ICC profile,
    a :exc:`PyCMSError` will be raised.

    :param profileFilename: String, as a valid filename path to the ICC profile
        you wish to open, or a file-like object.
    :returns: A CmsProfile class object.
    :exception PyCMSError:
    """
    try:
        return ImageCmsProfile(profileFilename)
    except (OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v