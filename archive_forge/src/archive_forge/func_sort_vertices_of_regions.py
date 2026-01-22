import numpy as np
import scipy
from . import _voronoi
from scipy.spatial import cKDTree
def sort_vertices_of_regions(self):
    """Sort indices of the vertices to be (counter-)clockwise ordered.

        Raises
        ------
        TypeError
            If the points are not three-dimensional.

        Notes
        -----
        For each region in regions, it sorts the indices of the Voronoi
        vertices such that the resulting points are in a clockwise or
        counterclockwise order around the generator point.

        This is done as follows: Recall that the n-th region in regions
        surrounds the n-th generator in points and that the k-th
        Voronoi vertex in vertices is the circumcenter of the k-th triangle
        in self._simplices.  For each region n, we choose the first triangle
        (=Voronoi vertex) in self._simplices and a vertex of that triangle
        not equal to the center n. These determine a unique neighbor of that
        triangle, which is then chosen as the second triangle. The second
        triangle will have a unique vertex not equal to the current vertex or
        the center. This determines a unique neighbor of the second triangle,
        which is then chosen as the third triangle and so forth. We proceed
        through all the triangles (=Voronoi vertices) belonging to the
        generator in points and obtain a sorted version of the vertices
        of its surrounding region.
        """
    if self._dim != 3:
        raise TypeError('Only supported for three-dimensional point sets')
    _voronoi.sort_vertices_of_regions(self._simplices, self.regions)