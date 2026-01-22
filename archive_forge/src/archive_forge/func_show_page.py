from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def show_page(self):
    """Emits and clears the current page
        for backends that support multiple pages.
        Use :meth:`copy_page` if you don't want to clear the page.

        This is a convenience method
        that simply calls :meth:`Surface.show_page`
        on the context’s target.

        """
    cairo.cairo_show_page(self._pointer)
    self._check_status()