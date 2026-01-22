from heapq import heappop, heappush
from shapely.errors import TopologicalError
from shapely.geometry import Point
def polylabel(polygon, tolerance=1.0):
    """Finds pole of inaccessibility for a given polygon. Based on
    Vladimir Agafonkin's https://github.com/mapbox/polylabel

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
    tolerance : int or float, optional
                `tolerance` represents the highest resolution in units of the
                input geometry that will be considered for a solution. (default
                value is 1.0).

    Returns
    -------
    shapely.geometry.Point
        A point representing the pole of inaccessibility for the given input
        polygon.

    Raises
    ------
    shapely.errors.TopologicalError
        If the input polygon is not a valid geometry.

    Example
    -------
    >>> from shapely import LineString
    >>> polygon = LineString([(0, 0), (50, 200), (100, 100), (20, 50),
    ... (-100, -20), (-150, -200)]).buffer(100)
    >>> polylabel(polygon, tolerance=10).wkt
    'POINT (59.35615556364569 121.83919629746435)'
    """
    if not polygon.is_valid:
        raise TopologicalError('Invalid polygon')
    minx, miny, maxx, maxy = polygon.bounds
    width = maxx - minx
    height = maxy - miny
    cell_size = min(width, height)
    h = cell_size / 2.0
    cell_queue = []
    x, y = polygon.centroid.coords[0]
    best_cell = Cell(x, y, 0, polygon)
    bbox_cell = Cell(minx + width / 2.0, miny + height / 2, 0, polygon)
    if bbox_cell.distance > best_cell.distance:
        best_cell = bbox_cell
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            heappush(cell_queue, Cell(x + h, y + h, h, polygon))
            y += cell_size
        x += cell_size
    while cell_queue:
        cell = heappop(cell_queue)
        if cell.distance > best_cell.distance:
            best_cell = cell
        if cell.max_distance - best_cell.distance <= tolerance:
            continue
        h = cell.h / 2.0
        heappush(cell_queue, Cell(cell.x - h, cell.y - h, h, polygon))
        heappush(cell_queue, Cell(cell.x + h, cell.y - h, h, polygon))
        heappush(cell_queue, Cell(cell.x - h, cell.y + h, h, polygon))
        heappush(cell_queue, Cell(cell.x + h, cell.y + h, h, polygon))
    return best_cell.centroid