from warnings import warn
import shapely
from shapely.algorithms.polylabel import polylabel  # noqa
from shapely.errors import GeometryTypeError, ShapelyDeprecationWarning
from shapely.geometry import (
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.geometry.polygon import orient as orient_
from shapely.prepared import prep
def voronoi_diagram(geom, envelope=None, tolerance=0.0, edges=False):
    """
    Constructs a Voronoi Diagram [1] from the given geometry.
    Returns a list of geometries.

    Parameters
    ----------
    geom: geometry
        the input geometry whose vertices will be used to calculate
        the final diagram.
    envelope: geometry, None
        clipping envelope for the returned diagram, automatically
        determined if None. The diagram will be clipped to the larger
        of this envelope or an envelope surrounding the sites.
    tolerance: float, 0.0
        sets the snapping tolerance used to improve the robustness
        of the computation. A tolerance of 0.0 specifies that no
        snapping will take place.
    edges: bool, False
        If False, return regions as polygons. Else, return only
        edges e.g. LineStrings.

    GEOS documentation can be found at [2]

    Returns
    -------
    GeometryCollection
        geometries representing the Voronoi regions.

    Notes
    -----
    The tolerance `argument` can be finicky and is known to cause the
    algorithm to fail in several cases. If you're using `tolerance`
    and getting a failure, try removing it. The test cases in
    tests/test_voronoi_diagram.py show more details.


    References
    ----------
    [1] https://en.wikipedia.org/wiki/Voronoi_diagram
    [2] https://geos.osgeo.org/doxygen/geos__c_8h_source.html  (line 730)
    """
    try:
        result = shapely.voronoi_polygons(geom, tolerance=tolerance, extend_to=envelope, only_edges=edges)
    except shapely.GEOSException as err:
        errstr = 'Could not create Voronoi Diagram with the specified inputs '
        errstr += f'({err!s}).'
        if tolerance:
            errstr += ' Try running again with default tolerance value.'
        raise ValueError(errstr) from err
    if result.geom_type != 'GeometryCollection':
        return GeometryCollection([result])
    return result