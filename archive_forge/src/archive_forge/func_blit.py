import numpy as np
from .. import cbook, transforms
from . import backend_agg, backend_gtk3
from .backend_gtk3 import Gtk, _BackendGTK3
import cairo  # Presence of cairo is already checked by _backend_gtk.
def blit(self, bbox=None):
    if bbox is None:
        bbox = self.figure.bbox
    scale = self.device_pixel_ratio
    allocation = self.get_allocation()
    x = int(bbox.x0 / scale)
    y = allocation.height - int(bbox.y1 / scale)
    width = (int(bbox.x1) - int(bbox.x0)) // scale
    height = (int(bbox.y1) - int(bbox.y0)) // scale
    self._bbox_queue.append(bbox)
    self.queue_draw_area(x, y, width, height)