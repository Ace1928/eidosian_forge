import copy
import gc
import pytest
import rpy2.rinterface as rinterface
def test_rid():
    globalenv_id = rinterface.baseenv.find('.GlobalEnv').rid
    assert globalenv_id == rinterface.globalenv.rid