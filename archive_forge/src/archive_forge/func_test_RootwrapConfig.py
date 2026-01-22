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
def test_RootwrapConfig(self):
    raw = configparser.RawConfigParser()
    self.assertRaises(configparser.Error, wrapper.RootwrapConfig, raw)
    raw.set('DEFAULT', 'filters_path', '/a,/b')
    config = wrapper.RootwrapConfig(raw)
    self.assertEqual(['/a', '/b'], config.filters_path)
    self.assertEqual(os.environ['PATH'].split(':'), config.exec_dirs)
    with fixtures.EnvironmentVariable('PATH'):
        c = wrapper.RootwrapConfig(raw)
        self.assertEqual([], c.exec_dirs)
    self.assertFalse(config.use_syslog)
    self.assertEqual(logging.handlers.SysLogHandler.LOG_SYSLOG, config.syslog_log_facility)
    self.assertEqual(logging.ERROR, config.syslog_log_level)
    raw.set('DEFAULT', 'exec_dirs', '/a,/x')
    config = wrapper.RootwrapConfig(raw)
    self.assertEqual(['/a', '/x'], config.exec_dirs)
    raw.set('DEFAULT', 'use_syslog', 'oui')
    self.assertRaises(ValueError, wrapper.RootwrapConfig, raw)
    raw.set('DEFAULT', 'use_syslog', 'true')
    config = wrapper.RootwrapConfig(raw)
    self.assertTrue(config.use_syslog)
    raw.set('DEFAULT', 'syslog_log_facility', 'moo')
    self.assertRaises(ValueError, wrapper.RootwrapConfig, raw)
    raw.set('DEFAULT', 'syslog_log_facility', 'local0')
    config = wrapper.RootwrapConfig(raw)
    self.assertEqual(logging.handlers.SysLogHandler.LOG_LOCAL0, config.syslog_log_facility)
    raw.set('DEFAULT', 'syslog_log_facility', 'LOG_AUTH')
    config = wrapper.RootwrapConfig(raw)
    self.assertEqual(logging.handlers.SysLogHandler.LOG_AUTH, config.syslog_log_facility)
    raw.set('DEFAULT', 'syslog_log_level', 'bar')
    self.assertRaises(ValueError, wrapper.RootwrapConfig, raw)
    raw.set('DEFAULT', 'syslog_log_level', 'INFO')
    config = wrapper.RootwrapConfig(raw)
    self.assertEqual(logging.INFO, config.syslog_log_level)