from unittest import TestCase
from traitlets import HasTraits, TraitError, observe, Undefined
from traitlets.tests.test_traitlets import TraitTestBase
from traittypes import Array, DataFrame, Series, Dataset, DataArray
import numpy as np
import pandas as pd
import xarray as xr
def test_initial_values(self):

    class Foo(HasTraits):
        b = DataArray(None, allow_none=True)
        c = DataArray([])
        d = DataArray(Undefined)
    foo = Foo()
    self.assertTrue(foo.b is None)
    self.assertTrue(foo.c.equals(xr.DataArray([])))
    self.assertTrue(foo.d is Undefined)