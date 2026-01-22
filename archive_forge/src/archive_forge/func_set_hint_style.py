from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
def set_hint_style(self, hint_style):
    """Changes the :ref:`HINT_STYLE` for the font options object.
        This controls whether to fit font outlines to the pixel grid,
        and if so, whether to optimize for fidelity or contrast.

        """
    cairo.cairo_font_options_set_hint_style(self._pointer, hint_style)
    self._check_status()