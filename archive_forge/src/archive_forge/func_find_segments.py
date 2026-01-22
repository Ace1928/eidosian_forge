from math import sqrt
from .gui import *
def find_segments(self, crossings, include_overcrossings=False):
    """
        Return a list of segments that make up this arrow, each
        segment being a list of 4 coordinates [x0,y0,x1,y1].  The
        first segment starts at the start vertex, and the last one
        ends at the end vertex.  Otherwise, endpoints are near
        crossings where this arrow goes under, leaving a gap between
        the endpoint and the crossing point.  If the
        include_overcrossings flag is True, then the segments are
        also split at overcrossings, with no gap.
        """
    params = self.params
    segments = []
    self.vectorize()
    cross_params = [(0.0, False), (1.0, False)]
    for c in crossings:
        if c.under == self:
            t = self ^ c.over
            if t:
                cross_params.append((t, not c.is_virtual))
        if c.over == self and (include_overcrossings or params['include_overcrossings']):
            t = self ^ c.under
            if t:
                cross_params.append((t, False))
    cross_params.sort()

    def r(t):
        """Affine parameterization of the arrow with domain [0,1]."""
        if t == 1.0:
            return list(self.end.point())
        x, y = self.start.point()
        return [x + t * self.dx, y + t * self.dy]
    segments = []
    for i in range(len(cross_params) - 1):
        a, has_gap_a = cross_params[i]
        b, has_gap_b = cross_params[i + 1]
        dt = b - a
        abs_gap = params['abs_gap_size'] / self.length if self.length != 0 else 0
        rel_gap = params['rel_gap_size'] * dt
        if params['double_gap_at_ends']:
            if i == 0 or i == len(cross_params) - 2:
                rel_gap = 2 * rel_gap
        gap = min(abs_gap, rel_gap)
        gap_a = gap if has_gap_a else 0
        gap_b = gap if has_gap_b else 0
        segments.append((a + gap_a, b - gap_b))
    return [r(a) + r(b) for a, b in segments]