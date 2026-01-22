from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testDefaultResetResetsDependencies(self):
    resource_manager = MockResettableResource()
    dep1 = MockResettableResource()
    dep2 = MockResettableResource()
    resource_manager.resources.append(('dep1', dep1))
    resource_manager.resources.append(('dep2', dep2))
    r_outer = resource_manager.getResource()
    r_inner = resource_manager.getResource()
    dep2.dirtied(r_inner.dep2)
    resource_manager.finishedWith(r_inner)
    r_inner = resource_manager.getResource()
    dep2.dirtied(r_inner.dep2)
    resource_manager.finishedWith(r_inner)
    resource_manager.finishedWith(r_outer)
    self.assertEqual(1, dep1.makes)
    self.assertEqual(1, dep1.cleans)
    self.assertEqual(0, dep1.resets)
    self.assertEqual(1, dep2.makes)
    self.assertEqual(1, dep2.cleans)
    self.assertEqual(1, dep2.resets)
    self.assertEqual(1, resource_manager.makes)
    self.assertEqual(1, resource_manager.cleans)
    self.assertEqual(1, resource_manager.resets)