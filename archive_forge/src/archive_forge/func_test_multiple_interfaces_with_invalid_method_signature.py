import unittest
import warnings
from traits.adaptation.api import reset_global_adaptation_manager
from traits.api import (
from traits.interface_checker import InterfaceError, check_implements
from traits import has_traits
def test_multiple_interfaces_with_invalid_method_signature(self):
    """ multiple interfaces with invalid method signature """

    class IFoo(Interface):

        def foo(self):
            pass

    class IBar(Interface):

        def bar(self):
            pass

    class IBaz(Interface):

        def baz(self):
            pass

    @provides(IFoo, IBar, IBaz)
    class Foo(HasTraits):

        def foo(self):
            pass

        def bar(self):
            pass

        def baz(self, x):
            pass
    self.assertRaises(InterfaceError, check_implements, Foo, [IFoo, IBar, IBaz], 2)