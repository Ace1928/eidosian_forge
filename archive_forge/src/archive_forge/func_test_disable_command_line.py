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
def test_disable_command_line(self):
    self.assertFalse(self.logs_present('parse_command_line()', ['--logging=none']))