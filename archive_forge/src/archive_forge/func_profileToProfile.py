from __future__ import annotations
import sys
from enum import IntEnum
from . import Image
def profileToProfile(im, inputProfile, outputProfile, renderingIntent=Intent.PERCEPTUAL, outputMode=None, inPlace=False, flags=0):
    """
    (pyCMS) Applies an ICC transformation to a given image, mapping from
    ``inputProfile`` to ``outputProfile``.

    If the input or output profiles specified are not valid filenames, a
    :exc:`PyCMSError` will be raised.  If ``inPlace`` is ``True`` and
    ``outputMode != im.mode``, a :exc:`PyCMSError` will be raised.
    If an error occurs during application of the profiles,
    a :exc:`PyCMSError` will be raised.
    If ``outputMode`` is not a mode supported by the ``outputProfile`` (or by pyCMS),
    a :exc:`PyCMSError` will be raised.

    This function applies an ICC transformation to im from ``inputProfile``'s
    color space to ``outputProfile``'s color space using the specified rendering
    intent to decide how to handle out-of-gamut colors.

    ``outputMode`` can be used to specify that a color mode conversion is to
    be done using these profiles, but the specified profiles must be able
    to handle that mode.  I.e., if converting im from RGB to CMYK using
    profiles, the input profile must handle RGB data, and the output
    profile must handle CMYK data.

    :param im: An open :py:class:`~PIL.Image.Image` object (i.e. Image.new(...)
        or Image.open(...), etc.)
    :param inputProfile: String, as a valid filename path to the ICC input
        profile you wish to use for this image, or a profile object
    :param outputProfile: String, as a valid filename path to the ICC output
        profile you wish to use for this image, or a profile object
    :param renderingIntent: Integer (0-3) specifying the rendering intent you
        wish to use for the transform

            ImageCms.Intent.PERCEPTUAL            = 0 (DEFAULT)
            ImageCms.Intent.RELATIVE_COLORIMETRIC = 1
            ImageCms.Intent.SATURATION            = 2
            ImageCms.Intent.ABSOLUTE_COLORIMETRIC = 3

        see the pyCMS documentation for details on rendering intents and what
        they do.
    :param outputMode: A valid PIL mode for the output image (i.e. "RGB",
        "CMYK", etc.).  Note: if rendering the image "inPlace", outputMode
        MUST be the same mode as the input, or omitted completely.  If
        omitted, the outputMode will be the same as the mode of the input
        image (im.mode)
    :param inPlace: Boolean.  If ``True``, the original image is modified in-place,
        and ``None`` is returned.  If ``False`` (default), a new
        :py:class:`~PIL.Image.Image` object is returned with the transform applied.
    :param flags: Integer (0-...) specifying additional flags
    :returns: Either None or a new :py:class:`~PIL.Image.Image` object, depending on
        the value of ``inPlace``
    :exception PyCMSError:
    """
    if outputMode is None:
        outputMode = im.mode
    if not isinstance(renderingIntent, int) or not 0 <= renderingIntent <= 3:
        msg = 'renderingIntent must be an integer between 0 and 3'
        raise PyCMSError(msg)
    if not isinstance(flags, int) or not 0 <= flags <= _MAX_FLAG:
        msg = f'flags must be an integer between 0 and {_MAX_FLAG}'
        raise PyCMSError(msg)
    try:
        if not isinstance(inputProfile, ImageCmsProfile):
            inputProfile = ImageCmsProfile(inputProfile)
        if not isinstance(outputProfile, ImageCmsProfile):
            outputProfile = ImageCmsProfile(outputProfile)
        transform = ImageCmsTransform(inputProfile, outputProfile, im.mode, outputMode, renderingIntent, flags=flags)
        if inPlace:
            transform.apply_in_place(im)
            imOut = None
        else:
            imOut = transform.apply(im)
    except (OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v
    return imOut