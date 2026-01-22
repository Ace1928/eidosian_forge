from unittest import mock
from oslotest import base as test_base
from testtools import matchers
from oslo_log import versionutils
@mock.patch('oslo_log.versionutils.report_deprecated_feature')
def test_deprecating_a_function_returns_correct_value(self, mock_reporter):

    @versionutils.deprecated(as_of=versionutils.deprecated.ICEHOUSE)
    def do_outdated_stuff(data):
        return data
    expected_rv = 'expected return value'
    retval = do_outdated_stuff(expected_rv)
    self.assertThat(retval, matchers.Equals(expected_rv))