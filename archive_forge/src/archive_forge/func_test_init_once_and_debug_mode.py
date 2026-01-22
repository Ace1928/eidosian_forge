import os
import sys
import logging
import tempfile
from unittest.mock import patch
import libcloud
from libcloud import _init_once
from libcloud.base import DriverTypeNotFoundError
from libcloud.test import unittest
from libcloud.utils.loggingconnection import LoggingConnection
def test_init_once_and_debug_mode(self):
    if have_paramiko:
        paramiko_logger = logging.getLogger('paramiko')
        paramiko_logger.setLevel(logging.INFO)
    _init_once()
    self.assertIsNone(LoggingConnection.log)
    if have_paramiko:
        paramiko_log_level = paramiko_logger.getEffectiveLevel()
        self.assertEqual(paramiko_log_level, logging.INFO)
    _, tmp_path = tempfile.mkstemp()
    os.environ['LIBCLOUD_DEBUG'] = tmp_path
    _init_once()
    self.assertTrue(LoggingConnection.log is not None)
    if have_paramiko:
        paramiko_log_level = paramiko_logger.getEffectiveLevel()
        self.assertEqual(paramiko_log_level, logging.DEBUG)