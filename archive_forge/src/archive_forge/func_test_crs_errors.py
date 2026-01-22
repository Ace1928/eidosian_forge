import numpy as np
from patsy.util import (have_pandas, atleast_2d_column_default,
from patsy.state import stateful_transform
def test_crs_errors():
    import pytest
    pytest.raises(ValueError, cr, np.arange(16).reshape((4, 4)), df=4)
    pytest.raises(ValueError, CR().transform, np.arange(16).reshape((4, 4)), df=4)
    pytest.raises(ValueError, cr, np.arange(50))
    pytest.raises(ValueError, cr, np.arange(50), df=4, constraints=np.arange(27).reshape((3, 3, 3)))
    pytest.raises(ValueError, cr, np.arange(50), df=4, constraints=np.arange(6))
    pytest.raises(ValueError, cr, np.arange(50), df=1)
    pytest.raises(ValueError, cc, np.arange(50), df=0)