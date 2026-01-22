from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def pop_group(self):
    """Terminates the redirection begun by a call to :meth:`push_group`
        or :meth:`push_group_with_content`
        and returns a new pattern containing the results
        of all drawing operations performed to the group.

        The :meth:`pop_group` method calls :meth:`restore`,
        (balancing a call to :meth:`save` by the push_group method),
        so that any changes to the graphics state
        will not be visible outside the group.

        :returns:
            A newly created :class:`SurfacePattern`
            containing the results of all drawing operations
            performed to the group.

        """
    return Pattern._from_pointer(cairo.cairo_pop_group(self._pointer), incref=False)