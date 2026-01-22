import pytest
from datashader import datashape
from datashader.datashape import dshape, has_var_dim, has_ellipsis
def test_cat_dshapes_errors():
    with pytest.raises(ValueError):
        datashape.cat_dshapes([])
    with pytest.raises(ValueError):
        datashape.cat_dshapes([dshape('3 * 10 * int32'), dshape('3 * 1 * int32')])