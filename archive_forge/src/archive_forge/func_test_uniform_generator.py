import pytest
import numpy
import geopandas
import geopandas._compat as compat
from geopandas.tools._random import uniform
@pytest.mark.skipif(not (compat.USE_PYGEOS or compat.USE_SHAPELY_20), reason='array input in interpolate not implemented for shapely<2')
def test_uniform_generator():
    sample = uniform(polygons[0], size=10, rng=1)
    sample2 = uniform(polygons[0], size=10, rng=1)
    assert sample.equals(sample2)
    generator = numpy.random.default_rng(seed=1)
    gen_sample = uniform(polygons[0], size=10, rng=generator)
    gen_sample2 = uniform(polygons[0], size=10, rng=generator)
    assert sample.equals(gen_sample)
    assert not sample.equals(gen_sample2)