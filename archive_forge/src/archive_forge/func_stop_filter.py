from contextlib import nullcontext
from math import radians, cos, sin
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib.backend_bases import (
from matplotlib.font_manager import fontManager as _fontManager, get_font
from matplotlib.ft2font import (LOAD_FORCE_AUTOHINT, LOAD_NO_HINTING,
from matplotlib.mathtext import MathTextParser
from matplotlib.path import Path
from matplotlib.transforms import Bbox, BboxBase
from matplotlib.backends._backend_agg import RendererAgg as _RendererAgg
def stop_filter(self, post_processing):
    """
        Save the current canvas as an image and apply post processing.

        The *post_processing* function::

           def post_processing(image, dpi):
             # ny, nx, depth = image.shape
             # image (numpy array) has RGBA channels and has a depth of 4.
             ...
             # create a new_image (numpy array of 4 channels, size can be
             # different). The resulting image may have offsets from
             # lower-left corner of the original image
             return new_image, offset_x, offset_y

        The saved renderer is restored and the returned image from
        post_processing is plotted (using draw_image) on it.
        """
    orig_img = np.asarray(self.buffer_rgba())
    slice_y, slice_x = cbook._get_nonzero_slices(orig_img[..., 3])
    cropped_img = orig_img[slice_y, slice_x]
    self._renderer = self._filter_renderers.pop()
    self._update_methods()
    if cropped_img.size:
        img, ox, oy = post_processing(cropped_img / 255, self.dpi)
        gc = self.new_gc()
        if img.dtype.kind == 'f':
            img = np.asarray(img * 255.0, np.uint8)
        self._renderer.draw_image(gc, slice_x.start + ox, int(self.height) - slice_y.stop + oy, img[::-1])