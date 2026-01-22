import pytest
import rpy2.robjects as robjects
import array
def test_init_invalid():
    with pytest.raises(ValueError):
        robjects.Environment('a')