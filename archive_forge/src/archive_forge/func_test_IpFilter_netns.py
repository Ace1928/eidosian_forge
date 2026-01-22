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
def test_IpFilter_netns(self):
    f = filters.IpFilter(self._ip, 'root')
    self.assertFalse(f.match(['ip', 'netns', 'exec', 'foo']))
    self.assertFalse(f.match(['ip', 'netns', 'exec']))
    self.assertFalse(f.match(['ip', '-s', 'netns', 'exec']))
    self.assertFalse(f.match(['ip', '-l', '42', 'netns', 'exec']))
    self.assertFalse(f.match(['ip', 'net', 'exec', 'foo']))
    self.assertFalse(f.match(['ip', 'netns', 'e', 'foo']))