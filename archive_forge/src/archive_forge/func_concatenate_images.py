import os
from glob import glob
import re
from collections.abc import Sequence
from copy import copy
import numpy as np
from PIL import Image
from tifffile import TiffFile
def concatenate_images(ic):
    """Concatenate all images in the image collection into an array.

    Parameters
    ----------
    ic : an iterable of images
        The images to be concatenated.

    Returns
    -------
    array_cat : ndarray
        An array having one more dimension than the images in `ic`.

    See Also
    --------
    ImageCollection.concatenate
    MultiImage.concatenate

    Raises
    ------
    ValueError
        If images in `ic` don't have identical shapes.

    Notes
    -----
    ``concatenate_images`` receives any iterable object containing images,
    including ImageCollection and MultiImage, and returns a NumPy array.
    """
    all_images = [image[np.newaxis, ...] for image in ic]
    try:
        array_cat = np.concatenate(all_images)
    except ValueError:
        raise ValueError('Image dimensions must agree.')
    return array_cat