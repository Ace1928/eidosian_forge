from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def tag_end(self, tag_name):
    """Marks the end of the ``tag_name`` structure.

        Invalid nesting of tags will cause @cr to shutdown with a status of
        ``CAIRO_STATUS_TAG_ERROR``.

        See :meth:`tag_begin`.

        :param tag_name: tag name

        *New in cairo 1.16.*

        *New in cairocffi 0.9.*

        """
    cairo.cairo_tag_end(self._pointer, _encode_string(tag_name))
    self._check_status()