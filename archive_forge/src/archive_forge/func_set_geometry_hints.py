import sys
import warnings
from collections import UserList
import gi
from gi.repository import GObject
def set_geometry_hints(self, geometry_widget=None, min_width=-1, min_height=-1, max_width=-1, max_height=-1, base_width=-1, base_height=-1, width_inc=-1, height_inc=-1, min_aspect=-1.0, max_aspect=-1.0):
    geometry = Gdk.Geometry()
    geom_mask = Gdk.WindowHints(0)
    if min_width >= 0 or min_height >= 0:
        geometry.min_width = max(min_width, 0)
        geometry.min_height = max(min_height, 0)
        geom_mask |= Gdk.WindowHints.MIN_SIZE
    if max_width >= 0 or max_height >= 0:
        geometry.max_width = max(max_width, 0)
        geometry.max_height = max(max_height, 0)
        geom_mask |= Gdk.WindowHints.MAX_SIZE
    if base_width >= 0 or base_height >= 0:
        geometry.base_width = max(base_width, 0)
        geometry.base_height = max(base_height, 0)
        geom_mask |= Gdk.WindowHints.BASE_SIZE
    if width_inc >= 0 or height_inc >= 0:
        geometry.width_inc = max(width_inc, 0)
        geometry.height_inc = max(height_inc, 0)
        geom_mask |= Gdk.WindowHints.RESIZE_INC
    if min_aspect >= 0.0 or max_aspect >= 0.0:
        if min_aspect <= 0.0 or max_aspect <= 0.0:
            raise TypeError('aspect ratios must be positive')
        geometry.min_aspect = min_aspect
        geometry.max_aspect = max_aspect
        geom_mask |= Gdk.WindowHints.ASPECT
    return orig_set_geometry_hints(self, geometry_widget, geometry, geom_mask)