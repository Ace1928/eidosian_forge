import sys
import unittest.mock
import warnings
import weakref
from traits.api import HasTraits
from traits.constants import (
from traits.ctrait import CTrait
from traits.trait_errors import TraitError
from traits.trait_types import Any, Int, List
def test_default_initialization(self):
    ctrait = CTrait()
    validate = unittest.mock.MagicMock(return_value='baz')
    ctrait.set_validate(validate)

    class Foo(HasTraits):
        bar = ctrait
        bar_changed = List

        def _bar_changed(self, new):
            self.bar_changed.append(new)
    foo = Foo()
    self.assertEqual(len(foo.bar_changed), 0)
    foo.bar = 1
    validate.assert_called_once_with(foo, 'bar', 1)
    self.assertEqual(foo.bar, 'baz')
    self.assertEqual(len(foo.bar_changed), 1)
    self.assertEqual(foo.bar_changed[0], 'baz')