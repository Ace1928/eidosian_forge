from __future__ import annotations
import functools
import operator
import re
from . import ExifTags, Image, ImagePalette
def exif_transpose(image, *, in_place=False):
    """
    If an image has an EXIF Orientation tag, other than 1, transpose the image
    accordingly, and remove the orientation data.

    :param image: The image to transpose.
    :param in_place: Boolean. Keyword-only argument.
        If ``True``, the original image is modified in-place, and ``None`` is returned.
        If ``False`` (default), a new :py:class:`~PIL.Image.Image` object is returned
        with the transposition applied. If there is no transposition, a copy of the
        image will be returned.
    """
    image.load()
    image_exif = image.getexif()
    orientation = image_exif.get(ExifTags.Base.Orientation)
    method = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180, 4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE, 6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE, 8: Image.Transpose.ROTATE_90}.get(orientation)
    if method is not None:
        transposed_image = image.transpose(method)
        if in_place:
            image.im = transposed_image.im
            image.pyaccess = None
            image._size = transposed_image._size
        exif_image = image if in_place else transposed_image
        exif = exif_image.getexif()
        if ExifTags.Base.Orientation in exif:
            del exif[ExifTags.Base.Orientation]
            if 'exif' in exif_image.info:
                exif_image.info['exif'] = exif.tobytes()
            elif 'Raw profile type exif' in exif_image.info:
                exif_image.info['Raw profile type exif'] = exif.tobytes().hex()
            elif 'XML:com.adobe.xmp' in exif_image.info:
                for pattern in ('tiff:Orientation="([0-9])"', '<tiff:Orientation>([0-9])</tiff:Orientation>'):
                    exif_image.info['XML:com.adobe.xmp'] = re.sub(pattern, '', exif_image.info['XML:com.adobe.xmp'])
        if not in_place:
            return transposed_image
    elif not in_place:
        return image.copy()