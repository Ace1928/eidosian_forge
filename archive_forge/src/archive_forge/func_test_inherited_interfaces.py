import unittest
import warnings
from traits.adaptation.api import reset_global_adaptation_manager
from traits.api import (
from traits.interface_checker import InterfaceError, check_implements
from traits import has_traits
def test_inherited_interfaces(self):
    """ inherited interfaces """

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
        z = Int
    check_implements(Foo, IBaz, 2)