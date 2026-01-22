from unittest import TestCase
from traitlets import HasTraits, TraitError, observe, Undefined
from traitlets.tests.test_traitlets import TraitTestBase
from traittypes import Array, DataFrame, Series, Dataset, DataArray
import numpy as np
import pandas as pd
import xarray as xr
def test_custom_validators(self):

    def squeeze(trait, value):
        if 1 in value.shape:
            value = np.squeeze(value)
        return value

    class Foo(HasTraits):
        bar = Array().valid(squeeze)
    foo = Foo(bar=[[1], [2]])
    self.assertTrue(np.array_equal(foo.bar, [1, 2]))
    foo.bar = [[1], [2], [3]]
    self.assertTrue(np.array_equal(foo.bar, [1, 2, 3]))

    def shape(*dimensions):

        def validator(trait, value):
            if value.shape != dimensions:
                raise TraitError('Expected an of shape %s and got and array with shape %s' % (dimensions, value.shape))
            else:
                return value
        return validator

    class Foo(HasTraits):
        bar = Array(np.identity(2)).valid(shape(2, 2))
    foo = Foo()
    with self.assertRaises(TraitError):
        foo.bar = [1]
    new_value = [[0, 1], [1, 0]]
    foo.bar = new_value
    self.assertTrue(np.array_equal(foo.bar, new_value))