from __future__ import annotations
import sys
from enum import IntEnum
from . import Image
def get_display_profile(handle=None):
    """
    (experimental) Fetches the profile for the current display device.

    :returns: ``None`` if the profile is not known.
    """
    if sys.platform != 'win32':
        return None
    from . import ImageWin
    if isinstance(handle, ImageWin.HDC):
        profile = core.get_display_profile_win32(handle, 1)
    else:
        profile = core.get_display_profile_win32(handle or 0)
    if profile is None:
        return None
    return ImageCmsProfile(profile)