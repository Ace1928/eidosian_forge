from unittest import mock
import betamax
from betamax import exceptions
import testtools
from keystoneauth1.fixture import keystoneauth_betamax
from keystoneauth1.fixture import serializer
from keystoneauth1.fixture import v2 as v2Fixtures
from keystoneauth1.identity import v2
from keystoneauth1 import session
@mock.patch.object(betamax.Betamax, 'register_serializer')
def test_can_pass_custom_serializer(self, register_serializer):
    serializer = mock.Mock()
    serializer.name = 'mocked-serializer'
    fixture = keystoneauth_betamax.BetamaxFixture(cassette_name='fake', cassette_library_dir='keystoneauth1/tests/unit/data', serializer=serializer)
    register_serializer.assert_called_once_with(serializer)
    self.assertIs(serializer, fixture.serializer)
    self.assertEqual('mocked-serializer', fixture.serializer_name)