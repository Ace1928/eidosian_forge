from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def clip_preserve(self):
    """Establishes a new clip region
        by intersecting the current clip region
        with the current path as it would be filled by :meth:`fill`
        and according to the current fill rule (see :meth:`set_fill_rule`).

        Unlike :meth:`clip`,
        :meth:`clip_preserve` preserves the path within the cairo context.

        The current clip region affects all drawing operations
        by effectively masking out any changes to the surface
        that are outside the current clip region.

        Calling :meth:`clip_preserve` can only make the clip region smaller,
        never larger.
        But the current clip is part of the graphics state,
        so a temporary restriction of the clip region can be achieved
        by calling :meth:`clip_preserve`
        within a :meth:`save` / :meth:`restore` pair.
        The only other means of increasing the size of the clip region
        is :meth:`reset_clip`.

        """
    cairo.cairo_clip_preserve(self._pointer)
    self._check_status()