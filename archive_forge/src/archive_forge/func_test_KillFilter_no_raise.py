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
def test_KillFilter_no_raise(self):
    """Makes sure ValueError from bug 926412 is gone."""
    f = filters.KillFilter('root', '')
    usercmd = ['notkill', 999999]
    self.assertFalse(f.match(usercmd))
    usercmd = ['kill', 'notapid']
    self.assertFalse(f.match(usercmd))
    self.assertFalse(f.match([]))
    self.assertFalse(f.match(None))