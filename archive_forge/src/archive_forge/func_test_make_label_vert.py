import pytest
from ...labels import (
def test_make_label_vert(self, args, multidim_sels, labellers):
    name, expected_label = args
    labeller_arg = labellers.labellers[name]
    label = labeller_arg.make_label_vert('theta', multidim_sels.sel, multidim_sels.isel)
    assert label == expected_label