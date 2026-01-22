import os
import testtools
from fixtures import EnvironmentVariable, TestWithFixtures
def test_cleanup_restores_when_missing(self):
    fixture = EnvironmentVariable('FIXTURES_TEST_VAR')
    os.environ['FIXTURES_TEST_VAR'] = 'bar'
    with fixture:
        os.environ.pop('FIXTURES_TEST_VAR', '')
    self.assertEqual('bar', os.environ.get('FIXTURES_TEST_VAR'))