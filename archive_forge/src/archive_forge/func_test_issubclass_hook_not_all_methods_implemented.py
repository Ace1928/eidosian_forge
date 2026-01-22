import abc
from neutron_lib.services import base
from neutron_lib.tests import _base as test_base
def test_issubclass_hook_not_all_methods_implemented(self):

    class A(TestPluginInterface.ServicePluginStub):

        def f(self):
            pass

    class B(base.ServicePluginBase):

        @abc.abstractmethod
        def f(self):
            pass

        @abc.abstractmethod
        def g(self):
            pass
    self.assertFalse(issubclass(A, B))