import pytest
from shapely import wkt
from shapely.errors import GEOSException
from shapely.geometry import LineString, Polygon, shape
from shapely.geos import geos_version
@pytest.mark.filterwarnings('ignore:Creating an ndarray from ragged nested sequences:')
@pytest.mark.parametrize('constructor, args', direct_cases)
def test_create_directly(constructor, args):
    with pytest.raises((ValueError, TypeError)) as exc:
        constructor(*args)
    assert exc.match("Inconsistent coordinate dimensionality|Input operand 0 does not have enough dimensions|ufunc 'linestrings' not supported for the input types|setting an array element with a sequence. The requested array has an inhomogeneous shape")