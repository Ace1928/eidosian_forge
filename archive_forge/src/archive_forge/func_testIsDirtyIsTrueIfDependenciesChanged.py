from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testIsDirtyIsTrueIfDependenciesChanged(self):
    resource_manager = MockResource()
    dep1 = MockResource()
    dep2 = MockResource()
    dep3 = MockResource()
    resource_manager.resources.append(('dep1', dep1))
    resource_manager.resources.append(('dep2', dep2))
    resource_manager.resources.append(('dep3', dep3))
    r = resource_manager.getResource()
    dep2.dirtied(r.dep2)
    r2 = dep2.getResource()
    self.assertTrue(resource_manager.isDirty())
    resource_manager.finishedWith(r)
    dep2.finishedWith(r2)