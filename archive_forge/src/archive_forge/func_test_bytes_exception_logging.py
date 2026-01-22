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
def test_bytes_exception_logging(self):
    try:
        raise Exception(b'\xe9')
    except Exception:
        self.logger.exception('caught exception')
    output = self.get_output()
    self.assertRegex(output, b'Exception.*\\\\xe9')
    self.assertNotIn(b'\\n', output)