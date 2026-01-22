from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
def set_hint_metrics(self, hint_metrics):
    """Changes the :ref:`HINT_METRICS` for the font options object.
        This controls whether metrics are quantized
        to integer values in device units.

        """
    cairo.cairo_font_options_set_hint_metrics(self._pointer, hint_metrics)
    self._check_status()