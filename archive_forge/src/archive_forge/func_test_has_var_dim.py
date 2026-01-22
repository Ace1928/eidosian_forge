import pytest
from datashader import datashape
from datashader.datashape import dshape, has_var_dim, has_ellipsis
@pytest.mark.parametrize('ds_pos', ['... * float32', 'A... * float32', 'var * float32', '10 * { f0: int32, f1: A... * float32 }', '{ f0 : { g0 : var * int }, f1: int32 }', (dshape('var * int32'),)])
def test_has_var_dim(ds_pos):
    assert has_var_dim(dshape(ds_pos))