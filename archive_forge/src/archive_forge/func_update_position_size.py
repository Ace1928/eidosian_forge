from __future__ import annotations
from typing import TYPE_CHECKING
from matplotlib import artist
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.text import _get_textbox  # type: ignore
from matplotlib.transforms import Affine2D
def update_position_size(self, renderer: RendererBase):
    """
        Update the location and the size of the bbox.
        """
    if self._update:
        return
    text = self.text
    posx, posy = text.get_transform().transform((text._x, text._y))
    x, y, w, h = _get_textbox(text, renderer)
    self.set_bounds(0.0, 0.0, w, h)
    self.set_transform(Affine2D().rotate_deg(text.get_rotation()).translate(posx + x, posy + y))
    fontsize_in_pixel = renderer.points_to_pixels(text.get_size())
    self.set_mutation_scale(fontsize_in_pixel)
    self._update = True