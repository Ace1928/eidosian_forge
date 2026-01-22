import mock # Yes, we only test the rolling backport
import testtools
from fixtures import (
def test_mock_patch_without_replacement(self):
    self.useFixture(MockPatchMultiple('%s.Foo' % __name__, bar=MockPatchMultiple.DEFAULT))
    instance = Foo()
    self.assertIsInstance(instance.bar(), mock.MagicMock)