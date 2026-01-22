import io
import math
import os
import typing
import weakref
def get_image_rects(page: fitz.Page, name, transform=False) -> list:
    """Return list of image positions on a page.

    Args:
        name: (str, list, int) image identification. May be reference name, an
              item of the page's image list or an xref.
        transform: (bool) whether to also return the transformation matrix.
    Returns:
        A list of fitz.Rect objects or tuples of (fitz.Rect, fitz.Matrix) for all image
        locations on the page.
    """
    if type(name) in (list, tuple):
        xref = name[0]
    elif type(name) is int:
        xref = name
    else:
        imglist = [i for i in page.get_images() if i[7] == name]
        if imglist == []:
            raise ValueError('bad image name')
        elif len(imglist) != 1:
            raise ValueError('multiple image names found')
        xref = imglist[0][0]
    pix = fitz.Pixmap(page.parent, xref)
    digest = pix.digest
    del pix
    infos = page.get_image_info(hashes=True)
    if not transform:
        bboxes = [fitz.Rect(im['bbox']) for im in infos if im['digest'] == digest]
    else:
        bboxes = [(fitz.Rect(im['bbox']), fitz.Matrix(im['transform'])) for im in infos if im['digest'] == digest]
    return bboxes