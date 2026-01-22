from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def test_dependencies_passed_to_factory(self):
    calls = []

    class Wrapped:

        def __init__(self, **args):
            calls.append(args)

        def setUp(self):
            pass

        def tearDown(self):
            pass

    class Trivial(testresources.TestResource):

        def __init__(self, thing):
            testresources.TestResource.__init__(self)
            self.thing = thing

        def make(self, dependency_resources):
            return self.thing

        def clean(self, resource):
            pass
    mgr = testresources.GenericResource(Wrapped)
    mgr.resources = [('foo', Trivial('foo')), ('bar', Trivial('bar'))]
    resource = mgr.getResource()
    self.assertEqual([{'foo': 'foo', 'bar': 'bar'}], calls)
    mgr.finishedWith(resource)