from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testneededResourcesDependenciesFirst(self):
    resource = testresources.TestResource()
    dep1 = testresources.TestResource()
    dep2 = testresources.TestResource()
    resource.resources.append(('dep1', dep1))
    resource.resources.append(('dep2', dep2))
    self.assertEqual([dep1, dep2, resource], resource.neededResources())