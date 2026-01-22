import mock # Yes, we only test the rolling backport
import testtools
from fixtures import (
def test_mock_patch_with_replacement(self):
    self.useFixture(MockPatch('%s.Foo.bar' % __name__, mocking_bar))
    instance = Foo()
    self.assertEqual(instance.bar(), 'mocked!')