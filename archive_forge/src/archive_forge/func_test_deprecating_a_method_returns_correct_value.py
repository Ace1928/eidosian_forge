from unittest import mock
from oslotest import base as test_base
from testtools import matchers
from oslo_log import versionutils
@mock.patch('oslo_log.versionutils.report_deprecated_feature')
def test_deprecating_a_method_returns_correct_value(self, mock_reporter):

    class C(object):

        @versionutils.deprecated(as_of=versionutils.deprecated.ICEHOUSE)
        def outdated_method(self, *args):
            return args
    retval = C().outdated_method(1, 'of anything')
    self.assertThat(retval, matchers.Equals((1, 'of anything')))