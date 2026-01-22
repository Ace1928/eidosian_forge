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
def test_EnvFilter_without_leading_env(self):
    envset = ['A=/some/thing', 'B=somethingelse']
    envcmd = ['env'] + envset
    realcmd = ['sleep', '10']
    f = filters.EnvFilter('sleep', 'root', 'A=', 'B=ignored')
    self.assertTrue(f.match(envset + ['sleep']))
    self.assertEqual(realcmd, f.get_command(envcmd + realcmd))
    self.assertEqual(realcmd, f.get_command(envset + realcmd))
    env = f.get_environment(envset + realcmd)
    self.assertEqual('/some/thing', env.get('A'))
    self.assertEqual('somethingelse', env.get('B'))
    self.assertNotIn('sleep', env.keys())