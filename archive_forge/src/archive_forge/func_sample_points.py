from warnings import warn
import warnings
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from shapely.geometry import box, MultiPoint
from shapely.geometry.base import BaseGeometry
from . import _compat as compat
from .array import GeometryArray, GeometryDtype, points_from_xy
def sample_points(self, size, method='uniform', seed=None, rng=None, **kwargs):
    """
        Sample points from each geometry.

        Generate a MultiPoint per each geometry containing points sampled from the
        geometry. You can either sample randomly from a uniform distribution or use an
        advanced sampling algorithm from the ``pointpats`` package.

        For polygons, this samples within the area of the polygon. For lines,
        this samples along the length of the linestring. For multi-part
        geometries, the weights of each part are selected according to their relevant
        attribute (area for Polygons, length for LineStrings), and then points are
        sampled from each part.

        Any other geometry type (e.g. Point, GeometryCollection) is ignored, and an
        empty MultiPoint geometry is returned.

        Parameters
        ----------
        size : int | array-like
            The size of the sample requested. Indicates the number of samples to draw
            from each geometry.  If an array of the same length as a GeoSeries is
            passed, it denotes the size of a sample per geometry.
        method : str, default "uniform"
            The sampling method. ``uniform`` samples uniformly at random from a
            geometry using ``numpy.random.uniform``. Other allowed strings
            (e.g. ``"cluster_poisson"``) denote sampling function name from the
            ``pointpats.random`` module (see
            http://pysal.org/pointpats/api.html#random-distributions). Pointpats methods
            are implemented for (Multi)Polygons only and will return an empty MultiPoint
            for other geometry types.
        rng : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
            A random generator or seed to initialize the numpy BitGenerator. If None, then fresh,
            unpredictable entropy will be pulled from the OS.
        **kwargs : dict
            Options for the pointpats sampling algorithms.

        Returns
        -------
        GeoSeries
            Points sampled within (or along) each geometry.

        Examples
        --------
        >>> from shapely.geometry import Polygon
        >>> s = geopandas.GeoSeries(
        ...     [
        ...         Polygon([(1, -1), (1, 0), (0, 0)]),
        ...         Polygon([(3, -1), (4, 0), (3, 1)]),
        ...     ]
        ... )

        >>> s.sample_points(size=10)  # doctest: +SKIP
        0    MULTIPOINT (0.04783 -0.04244, 0.24196 -0.09052...
        1    MULTIPOINT (3.00672 -0.52390, 3.01776 0.30065,...
        Name: sampled_points, dtype: geometry
        """
    from .geoseries import GeoSeries
    from .tools._random import uniform
    if seed is not None:
        warn("The 'seed' keyword is deprecated. Use 'rng' instead.", FutureWarning, stacklevel=2)
        rng = seed
    if method == 'uniform':
        if pd.api.types.is_list_like(size):
            result = [uniform(geom, s, rng) for geom, s in zip(self.geometry, size)]
        else:
            result = self.geometry.apply(uniform, size=size, rng=rng)
    else:
        pointpats = compat.import_optional_dependency('pointpats', f"For complex sampling methods, the pointpats module is required. Your requested method, '{method}' was not a supported option and the pointpats package was not able to be imported.")
        if not hasattr(pointpats.random, method):
            raise AttributeError(f'pointpats.random module has no sampling method {method}.Consult the pointpats.random module documentation for available random sampling methods.')
        sample_function = getattr(pointpats.random, method)
        result = self.geometry.apply(lambda x: points_from_xy(*sample_function(x, size=size, **kwargs).T).unary_union() if not (x.is_empty or x is None or 'Polygon' not in x.geom_type) else MultiPoint())
    return GeoSeries(result, name='sampled_points', crs=self.crs, index=self.index)