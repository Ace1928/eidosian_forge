import logging
import numpy as np
from .pillow_legacy import PillowFormat, image_as_uint, ndarray_to_pil
def getSubRectangle(self, im):
    """Calculate the minimal rectangle that need updating. Returns
        a two-element tuple containing the cropped image and an x-y tuple.

        Calculating the subrectangles takes extra time, obviously. However,
        if the image sizes were reduced, the actual writing of the GIF
        goes faster. In some cases applying this method produces a GIF faster.
        """
    if self._count == 0:
        return (im, (0, 0))
    prev = self._previous_image
    diff = np.abs(im - prev)
    if diff.ndim == 3:
        diff = diff.sum(2)
    X = np.argwhere(diff.sum(0))
    Y = np.argwhere(diff.sum(1))
    if X.size and Y.size:
        x0, x1 = (int(X[0]), int(X[-1] + 1))
        y0, y1 = (int(Y[0]), int(Y[-1] + 1))
    else:
        x0, x1 = (0, 2)
        y0, y1 = (0, 2)
    return (im[y0:y1, x0:x1], (x0, y0))