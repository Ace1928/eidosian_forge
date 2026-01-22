import sys
import warnings
from functools import partial
from textwrap import indent
import pytest
from nibabel.deprecator import (
from ..testing import clear_and_catch_warnings
def test_deprecator_maker(self):
    dec = self.dep_maker(warn_class=UserWarning)
    func = dec('foo')(func_no_doc)
    with pytest.warns(UserWarning) as w:
        assert func() is None
        assert len(w) == 1
    dec = self.dep_maker(error_class=CustomError)
    func = dec('foo')(func_no_doc)
    with pytest.deprecated_call():
        assert func() is None
    func = dec('foo', until='1.8')(func_no_doc)
    with pytest.raises(CustomError):
        func()