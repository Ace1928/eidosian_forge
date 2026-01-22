from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testneededResourcesDefault(self):
    resource = testresources.TestResource()
    self.assertEqual([resource], resource.neededResources())