import geopandas
import pytest
def test_options_io_engine():
    assert geopandas.options.io_engine is None
    geopandas.options.io_engine = 'pyogrio'
    assert geopandas.options.io_engine == 'pyogrio'
    with pytest.raises(ValueError):
        geopandas.options.io_engine = 'abc'
    with pytest.raises(ValueError):
        geopandas.options.io_engine = -1
    geopandas.options.io_engine = None