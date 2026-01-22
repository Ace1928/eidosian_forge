from unittest import mock
import betamax
from betamax import exceptions
import testtools
from keystoneauth1.fixture import keystoneauth_betamax
from keystoneauth1.fixture import serializer
from keystoneauth1.fixture import v2 as v2Fixtures
from keystoneauth1.identity import v2
from keystoneauth1 import session
def test_no_serializer_options_provided(self):
    fixture = keystoneauth_betamax.BetamaxFixture(cassette_name='fake', cassette_library_dir='keystoneauth1/tests/unit/data')
    self.assertIs(serializer.YamlJsonSerializer, fixture.serializer)
    self.assertEqual('yamljson', fixture.serializer_name)