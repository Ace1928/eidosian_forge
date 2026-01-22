import os
import testtools
from fixtures import EnvironmentVariable, TestWithFixtures
def test_setup_deletes(self):
    fixture = EnvironmentVariable('FIXTURES_TEST_VAR')
    os.environ['FIXTURES_TEST_VAR'] = 'foo'
    self.useFixture(fixture)
    self.assertEqual(None, os.environ.get('FIXTURES_TEST_VAR'))