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
def test_missing_command(self):
    valid_but_missing = ['foo_bar_not_exist']
    invalid = ['foo_bar_not_exist_and_not_matched']
    self.assertRaises(wrapper.FilterMatchNotExecutable, wrapper.match_filter, self.filters, valid_but_missing)
    self.assertRaises(wrapper.NoFilterMatched, wrapper.match_filter, self.filters, invalid)