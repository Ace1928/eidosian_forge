import time
from string import ascii_lowercase
from .gui import tkMessageBox
from .vertex import Vertex
from .arrow import Arrow, default_arrow_params
from .crossings import Crossing, ECrossing
from .smooth import TikZPicture
def polylines(self, break_at_overcrossings=True):
    """
        Returns a list of lists of polylines, one per component, that make up
        the drawing of the link diagram.  Each polyline is a maximal
        segment with no undercrossings (e.g. corresponds to a generator
        in the Wirtinger presentation).  Each polyline is a list of
        coordinates [(x0,y0), (x1,y1), ...]  Isolated vertices are
        ignored.

        If the flag break_at_overcrossings is set, each polyline instead
        corresponds to maximal arcs with no crossings on their interior.
        """
    result = []
    self.update_crosspoints()
    segments = {}
    for arrow in self.Arrows:
        arrows_segments = arrow.find_segments(self.Crossings, include_overcrossings=True)
        segments[arrow] = [[(x0, y0), (x1, y1)] for x0, y0, x1, y1 in arrows_segments]
    if break_at_overcrossings:
        crossing_locations = set([(c.x, c.y) for c in self.Crossings])
    for component in self.arrow_components():
        color = component[0].color
        polylines = []
        polyline = []
        for arrow in component:
            for segment in segments[arrow]:
                if len(polyline) == 0:
                    polyline = segment
                elif segment[0] == polyline[-1]:
                    if break_at_overcrossings and segment[0] in crossing_locations:
                        polylines.append(polyline)
                        polyline = segment
                    else:
                        polyline.append(segment[1])
                else:
                    polylines.append(polyline)
                    polyline = segment
        polylines.append(polyline)
        if polylines[0][0] == polylines[-1][-1]:
            if len(polylines) > 1:
                polylines[0] = polylines.pop()[:-1] + polylines[0]
        result.append((polylines, color))
    return result