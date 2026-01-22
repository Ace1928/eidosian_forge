from __future__ import annotations
import sys
from enum import IntEnum
from . import Image
def getDefaultIntent(profile):
    """
    (pyCMS) Gets the default intent name for the given profile.

    If ``profile`` isn't a valid CmsProfile object or filename to a profile, a
    :exc:`PyCMSError` is raised.

    If an error occurs while trying to obtain the default intent, a
    :exc:`PyCMSError` is raised.

    Use this function to determine the default (and usually best optimized)
    rendering intent for this profile.  Most profiles support multiple
    rendering intents, but are intended mostly for one type of conversion.
    If you wish to use a different intent than returned, use
    ImageCms.isIntentSupported() to verify it will work first.

    :param profile: EITHER a valid CmsProfile object, OR a string of the
        filename of an ICC profile.
    :returns: Integer 0-3 specifying the default rendering intent for this
        profile.

            ImageCms.Intent.PERCEPTUAL            = 0 (DEFAULT)
            ImageCms.Intent.RELATIVE_COLORIMETRIC = 1
            ImageCms.Intent.SATURATION            = 2
            ImageCms.Intent.ABSOLUTE_COLORIMETRIC = 3

        see the pyCMS documentation for details on rendering intents and what
            they do.
    :exception PyCMSError:
    """
    try:
        if not isinstance(profile, ImageCmsProfile):
            profile = ImageCmsProfile(profile)
        return profile.profile.rendering_intent
    except (AttributeError, OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v