import array
import pytest
import rpy2.robjects as robjects
def test_rclass():
    assert robjects.baseenv['letters'].rclass[0] == 'character'
    assert robjects.baseenv['pi'].rclass[0] == 'numeric'
    assert robjects.globalenv.find('help').rclass[0] == 'function'