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
def test_IpFilter_non_netns(self):
    f = filters.IpFilter(self._ip, 'root')
    self.assertTrue(f.match(['ip', 'link', 'list']))
    self.assertTrue(f.match(['ip', '-s', 'link', 'list']))
    self.assertTrue(f.match(['ip', '-s', '-v', 'netns', 'add']))
    self.assertTrue(f.match(['ip', 'link', 'set', 'interface', 'netns', 'somens']))