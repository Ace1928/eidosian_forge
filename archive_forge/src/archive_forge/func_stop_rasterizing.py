import numpy as np
from matplotlib import cbook
from .backend_agg import RendererAgg
from matplotlib._tight_bbox import process_figure_for_rasterizing
def stop_rasterizing(self):
    """
        Exit "raster" mode.  All of the drawing that was done since
        the last `start_rasterizing` call will be copied to the
        vector backend by calling draw_image.
        """
    self._renderer = self._vector_renderer
    height = self._height * self.dpi
    img = np.asarray(self._raster_renderer.buffer_rgba())
    slice_y, slice_x = cbook._get_nonzero_slices(img[..., 3])
    cropped_img = img[slice_y, slice_x]
    if cropped_img.size:
        gc = self._renderer.new_gc()
        self._renderer.draw_image(gc, slice_x.start * self._figdpi / self.dpi, (height - slice_y.stop) * self._figdpi / self.dpi, cropped_img[::-1])
    self._raster_renderer = None
    self.figure.dpi = self._figdpi
    if self._bbox_inches_restore:
        r = process_figure_for_rasterizing(self.figure, self._bbox_inches_restore, self._figdpi)
        self._bbox_inches_restore = r