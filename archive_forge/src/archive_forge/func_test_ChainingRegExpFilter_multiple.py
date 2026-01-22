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
def test_ChainingRegExpFilter_multiple(self):
    filter_list = [filters.ChainingRegExpFilter('ionice', 'root', 'ionice', '-c[0-3]'), filters.ChainingRegExpFilter('ionice', 'root', 'ionice', '-c[0-3]', '-n[0-7]'), filters.CommandFilter('cat', 'root')]
    args = ['ionice', '-c2', '-n7', 'cat', '/a']
    dirs = ['/bin', '/usr/bin']
    self.assertIsNotNone(wrapper.match_filter(filter_list, args, dirs))