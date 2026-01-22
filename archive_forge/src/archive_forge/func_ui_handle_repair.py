from __future__ import annotations
from . import Image
def ui_handle_repair(self, dc, x0, y0, x1, y1):
    self.image.draw(dc, (x0, y0, x1, y1))