import configparser
import logging
import logging.handlers
import os
import tempfile
from unittest import mock
import uuid
import fixtures
import testtools
from oslo_rootwrap import cmd
from oslo_rootwrap import daemon
from oslo_rootwrap import filters
from oslo_rootwrap import subprocess
from oslo_rootwrap import wrapper
def test_match_filter_recurses_exec_command_filter_matches(self):
    filter_list = [filters.IpNetnsExecFilter(self._ip, 'root'), filters.IpFilter(self._ip, 'root')]
    args = ['ip', 'netns', 'exec', 'foo', 'ip', 'link', 'list']
    self.assertIsNotNone(wrapper.match_filter(filter_list, args))