import numpy
import pytest
from hypothesis import given, settings
from hypothesis.strategies import composite, integers, lists
from numpy.testing import assert_allclose
from thinc.layers import Embed
from thinc.layers.uniqued import uniqued
@given(X=lists_of_integers(lo=0, hi=ROWS - 1))
@settings(deadline=None)
def test_uniqued_doesnt_change_result(model, X):
    umodel = uniqued(model, column=model.attrs['column']).initialize()
    Y, bp_Y = model(X, is_train=True)
    Yu, bp_Yu = umodel(X, is_train=True)
    assert_allclose(Y, Yu)
    dX = bp_Y(Y)
    dXu = bp_Yu(Yu)
    assert_allclose(dX, dXu)
    if X.size:
        pass