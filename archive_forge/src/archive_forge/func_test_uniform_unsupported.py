import pytest
import numpy
import geopandas
import geopandas._compat as compat
from geopandas.tools._random import uniform
@pytest.mark.skipif(not (compat.USE_PYGEOS or compat.USE_SHAPELY_20), reason='array input in interpolate not implemented for shapely<2')
def test_uniform_unsupported():
    with pytest.warns(UserWarning, match='Sampling is not supported'):
        sample = uniform(points[0], size=10, rng=1)
    assert sample.is_empty