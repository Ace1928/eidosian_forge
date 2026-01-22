import testtools
from unittest import mock
from aodhclient import exceptions
from aodhclient.v2 import quota_cli
def test_quota_set_invalid_quota(self):
    parser = self.quota_set.get_parser('')
    args = parser.parse_args(['fake_project', '--alarm', '-2'])
    self.assertRaises(exceptions.CommandError, self.quota_set.take_action, args)