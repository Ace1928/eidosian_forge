import contextlib
import glob
import logging
import os
import re
import subprocess
import sys
import tempfile
import unittest
import warnings
from tornado.escape import utf8
from tornado.log import LogFormatter, define_logging_options, enable_pretty_logging
from tornado.options import OptionParser
from tornado.util import basestring_type
def logs_present(self, statement, args=None):
    IMPORT = 'from tornado.options import options, parse_command_line'
    LOG_INFO = 'import logging; logging.info("hello")'
    program = ';'.join([IMPORT, statement, LOG_INFO])
    proc = subprocess.Popen([sys.executable, '-c', program] + (args or []), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = proc.communicate()
    self.assertEqual(proc.returncode, 0, 'process failed: %r' % stdout)
    return b'hello' in stdout