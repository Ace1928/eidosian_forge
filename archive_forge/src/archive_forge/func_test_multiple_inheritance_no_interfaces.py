import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_multiple_inheritance_no_interfaces(self):
    from zope.interface.declarations import implementedBy
    from zope.interface.declarations import implementer
    from zope.interface.interface import Interface

    class IDefaultViewName(Interface):
        pass

    class Context:
        pass

    class RDBModel(Context):
        pass

    class IOther(Interface):
        pass

    @implementer(IOther)
    class OtherBase:
        pass

    class Model(OtherBase, Context):
        pass
    self.assertEqual(implementedBy(Model).__sro__, (implementedBy(Model), implementedBy(OtherBase), IOther, implementedBy(Context), implementedBy(object), Interface))