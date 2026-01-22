from __future__ import annotations
import sys
from enum import IntEnum
from . import Image
def buildTransform(inputProfile, outputProfile, inMode, outMode, renderingIntent=Intent.PERCEPTUAL, flags=0):
    """
    (pyCMS) Builds an ICC transform mapping from the ``inputProfile`` to the
    ``outputProfile``. Use applyTransform to apply the transform to a given
    image.

    If the input or output profiles specified are not valid filenames, a
    :exc:`PyCMSError` will be raised. If an error occurs during creation
    of the transform, a :exc:`PyCMSError` will be raised.

    If ``inMode`` or ``outMode`` are not a mode supported by the ``outputProfile``
    (or by pyCMS), a :exc:`PyCMSError` will be raised.

    This function builds and returns an ICC transform from the ``inputProfile``
    to the ``outputProfile`` using the ``renderingIntent`` to determine what to do
    with out-of-gamut colors.  It will ONLY work for converting images that
    are in ``inMode`` to images that are in ``outMode`` color format (PIL mode,
    i.e. "RGB", "RGBA", "CMYK", etc.).

    Building the transform is a fair part of the overhead in
    ImageCms.profileToProfile(), so if you're planning on converting multiple
    images using the same input/output settings, this can save you time.
    Once you have a transform object, it can be used with
    ImageCms.applyProfile() to convert images without the need to re-compute
    the lookup table for the transform.

    The reason pyCMS returns a class object rather than a handle directly
    to the transform is that it needs to keep track of the PIL input/output
    modes that the transform is meant for.  These attributes are stored in
    the ``inMode`` and ``outMode`` attributes of the object (which can be
    manually overridden if you really want to, but I don't know of any
    time that would be of use, or would even work).

    :param inputProfile: String, as a valid filename path to the ICC input
        profile you wish to use for this transform, or a profile object
    :param outputProfile: String, as a valid filename path to the ICC output
        profile you wish to use for this transform, or a profile object
    :param inMode: String, as a valid PIL mode that the appropriate profile
        also supports (i.e. "RGB", "RGBA", "CMYK", etc.)
    :param outMode: String, as a valid PIL mode that the appropriate profile
        also supports (i.e. "RGB", "RGBA", "CMYK", etc.)
    :param renderingIntent: Integer (0-3) specifying the rendering intent you
        wish to use for the transform

            ImageCms.Intent.PERCEPTUAL            = 0 (DEFAULT)
            ImageCms.Intent.RELATIVE_COLORIMETRIC = 1
            ImageCms.Intent.SATURATION            = 2
            ImageCms.Intent.ABSOLUTE_COLORIMETRIC = 3

        see the pyCMS documentation for details on rendering intents and what
        they do.
    :param flags: Integer (0-...) specifying additional flags
    :returns: A CmsTransform class object.
    :exception PyCMSError:
    """
    if not isinstance(renderingIntent, int) or not 0 <= renderingIntent <= 3:
        msg = 'renderingIntent must be an integer between 0 and 3'
        raise PyCMSError(msg)
    if not isinstance(flags, int) or not 0 <= flags <= _MAX_FLAG:
        msg = 'flags must be an integer between 0 and %s' + _MAX_FLAG
        raise PyCMSError(msg)
    try:
        if not isinstance(inputProfile, ImageCmsProfile):
            inputProfile = ImageCmsProfile(inputProfile)
        if not isinstance(outputProfile, ImageCmsProfile):
            outputProfile = ImageCmsProfile(outputProfile)
        return ImageCmsTransform(inputProfile, outputProfile, inMode, outMode, renderingIntent, flags=flags)
    except (OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v