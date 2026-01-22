from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def set_dash(self, dashes, offset=0):
    """Sets the dash pattern to be used by :meth:`stroke`.
        A dash pattern is specified by dashes, a list of positive values.
        Each value provides the length of alternate "on" and "off"
        portions of the stroke.
        ``offset`` specifies an offset into the pattern
        at which the stroke begins.

        Each "on" segment will have caps applied
        as if the segment were a separate sub-path.
        In particular, it is valid to use an "on" length of 0
        with :obj:`LINE_CAP_ROUND` or :obj:`LINE_CAP_SQUARE`
        in order to distributed dots or squares along a path.

        Note: The length values are in user-space units
        as evaluated at the time of stroking.
        This is not necessarily the same as the user space
        at the time of :meth:`set_dash`.

        If ``dashes`` is empty dashing is disabled.
        If it is of length 1 a symmetric pattern is assumed
        with alternating on and off portions of the size specified
        by the single value.

        :param dashes:
            A list of floats specifying alternate lengths
            of on and off stroke portions.
        :type offset: float
        :param offset:
            An offset into the dash pattern at which the stroke should start.
        :raises:
            :exc:`CairoError`
            if any value in dashes is negative,
            or if all values are 0.
            The context  will be put into an error state.

        """
    cairo.cairo_set_dash(self._pointer, ffi.new('double[]', dashes), len(dashes), offset)
    self._check_status()