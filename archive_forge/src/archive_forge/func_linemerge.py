from warnings import warn
import shapely
from shapely.algorithms.polylabel import polylabel  # noqa
from shapely.errors import GeometryTypeError, ShapelyDeprecationWarning
from shapely.geometry import (
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.geometry.polygon import orient as orient_
from shapely.prepared import prep
def linemerge(self, lines, directed=False):
    """Merges all connected lines from a source

        The source may be a MultiLineString, a sequence of LineString objects,
        or a sequence of objects than can be adapted to LineStrings.  Returns a
        LineString or MultiLineString when lines are not contiguous.
        """
    source = None
    if getattr(lines, 'geom_type', None) == 'MultiLineString':
        source = lines
    elif hasattr(lines, 'geoms'):
        source = MultiLineString([ls.coords for ls in lines.geoms])
    elif hasattr(lines, '__iter__'):
        try:
            source = MultiLineString([ls.coords for ls in lines])
        except AttributeError:
            source = MultiLineString(lines)
    if source is None:
        raise ValueError(f'Cannot linemerge {lines}')
    return shapely.line_merge(source, directed=directed)