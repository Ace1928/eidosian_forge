import unittest
import warnings
from traits.adaptation.api import reset_global_adaptation_manager
from traits.api import (
from traits.interface_checker import InterfaceError, check_implements
from traits import has_traits
def test_inherited_interfaces_with_missing_trait(self):
    """ inherited interfaces with missing trait """

    class IFoo(Interface):
        x = Int

    class IBar(IFoo):
        y = Int

    class IBaz(IBar):
        z = Int

    @provides(IBaz)
    class Foo(HasTraits):
        x = Int
        y = Int
    self.assertRaises(InterfaceError, check_implements, Foo, IBaz, 2)