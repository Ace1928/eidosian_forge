import io
import math
import os
import typing
import weakref
def insert_image(page, rect, *, alpha=-1, filename=None, height=0, keep_proportion=True, mask=None, oc=0, overlay=True, pixmap=None, rotate=0, stream=None, width=0, xref=0):
    """Insert an image for display in a rectangle.

    Args:
        rect: (rect_like) position of image on the page.
        alpha: (int, optional) set to 0 if image has no transparency.
        filename: (str, Path, file object) image filename.
        height: (int)
        keep_proportion: (bool) keep width / height ratio (default).
        mask: (bytes, optional) image consisting of alpha values to use.
        oc: (int) xref of OCG or OCMD to declare as Optional Content.
        overlay: (bool) put in foreground (default) or background.
        pixmap: (fitz.Pixmap) use this as image.
        rotate: (int) rotate by 0, 90, 180 or 270 degrees.
        stream: (bytes) use this as image.
        width: (int)
        xref: (int) use this as image.

    'page' and 'rect' are positional, all other parameters are keywords.

    If 'xref' is given, that image is used. Other input options are ignored.
    Else, exactly one of pixmap, stream or filename must be given.

    'alpha=0' for non-transparent images improves performance significantly.
    Affects stream and filename only.

    Optimum transparent insertions are possible by using filename / stream in
    conjunction with a 'mask' image of alpha values.

    Returns:
        xref (int) of inserted image. Re-use as argument for multiple insertions.
    """
    fitz.CheckParent(page)
    doc = page.parent
    if not doc.is_pdf:
        raise ValueError('is no PDF')
    if xref == 0 and bool(filename) + bool(stream) + bool(pixmap) != 1:
        raise ValueError('xref=0 needs exactly one of filename, pixmap, stream')
    if filename:
        if type(filename) is str:
            pass
        elif hasattr(filename, 'absolute'):
            filename = str(filename)
        elif hasattr(filename, 'name'):
            filename = filename.name
        else:
            raise ValueError('bad filename')
    if filename and (not os.path.exists(filename)):
        raise FileNotFoundError("No such file: '%s'" % filename)
    elif stream and type(stream) not in (bytes, bytearray, io.BytesIO):
        raise ValueError('stream must be bytes-like / BytesIO')
    elif pixmap and type(pixmap) is not fitz.Pixmap:
        raise ValueError('pixmap must be a fitz.Pixmap')
    if mask and (not (stream or filename)):
        raise ValueError('mask requires stream or filename')
    if mask and type(mask) not in (bytes, bytearray, io.BytesIO):
        raise ValueError('mask must be bytes-like / BytesIO')
    while rotate < 0:
        rotate += 360
    while rotate >= 360:
        rotate -= 360
    if rotate not in (0, 90, 180, 270):
        raise ValueError('bad rotate value')
    r = fitz.Rect(rect)
    if r.is_empty or r.is_infinite:
        raise ValueError('rect must be finite and not empty')
    clip = r * ~page.transformation_matrix
    ilst = [i[7] for i in doc.get_page_images(page.number)]
    ilst += [i[1] for i in doc.get_page_xobjects(page.number)]
    ilst += [i[4] for i in doc.get_page_fonts(page.number)]
    n = 'fzImg'
    i = 0
    _imgname = n + '0'
    while _imgname in ilst:
        i += 1
        _imgname = n + str(i)
    if overlay:
        page.wrap_contents()
    digests = doc.InsertedImages
    xref, digests = page._insert_image(filename=filename, pixmap=pixmap, stream=stream, imask=mask, clip=clip, overlay=overlay, oc=oc, xref=xref, rotate=rotate, keep_proportion=keep_proportion, width=width, height=height, alpha=alpha, _imgname=_imgname, digests=digests)
    if digests is not None:
        doc.InsertedImages = digests
    return xref