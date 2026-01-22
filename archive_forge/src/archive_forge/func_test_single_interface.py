import unittest
import warnings
from traits.adaptation.api import reset_global_adaptation_manager
from traits.api import (
from traits.interface_checker import InterfaceError, check_implements
from traits import has_traits
def test_single_interface(self):
    """ single interface """

    class IFoo(Interface):
        x = Int

    @provides(IFoo)
    class Foo(HasTraits):
        x = Int
    check_implements(Foo, IFoo, 2)