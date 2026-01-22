import datetime
import math
import os
import random
import re
import subprocess
import sys
import time
import types
import unittest
import warnings
from humanfriendly import (
from humanfriendly.case import CaseInsensitiveDict, CaseInsensitiveKey
from humanfriendly.cli import main
from humanfriendly.compat import StringIO
from humanfriendly.decorators import cached
from humanfriendly.deprecation import DeprecationProxy, define_aliases, deprecated_args, get_aliases
from humanfriendly.prompts import (
from humanfriendly.sphinx import (
from humanfriendly.tables import (
from humanfriendly.terminal import (
from humanfriendly.terminal.html import html_to_ansi
from humanfriendly.terminal.spinners import AutomaticSpinner, Spinner
from humanfriendly.testing import (
from humanfriendly.text import (
from humanfriendly.usage import (
from mock import MagicMock
def test_parse_timespan(self):
    """Test :func:`humanfriendly.parse_timespan()`."""
    self.assertEqual(0, parse_timespan('0'))
    self.assertEqual(0, parse_timespan('0s'))
    self.assertEqual(1e-09, parse_timespan('1ns'))
    self.assertEqual(5.1e-08, parse_timespan('51ns'))
    self.assertEqual(1e-06, parse_timespan('1us'))
    self.assertEqual(5.2e-05, parse_timespan('52us'))
    self.assertEqual(0.001, parse_timespan('1ms'))
    self.assertEqual(0.001, parse_timespan('1 millisecond'))
    self.assertEqual(0.5, parse_timespan('500 milliseconds'))
    self.assertEqual(0.5, parse_timespan('0.5 seconds'))
    self.assertEqual(5, parse_timespan('5s'))
    self.assertEqual(5, parse_timespan('5 seconds'))
    self.assertEqual(60 * 2, parse_timespan('2m'))
    self.assertEqual(60 * 2, parse_timespan('2 minutes'))
    self.assertEqual(60 * 3, parse_timespan('3 min'))
    self.assertEqual(60 * 3, parse_timespan('3 mins'))
    self.assertEqual(60 * 60 * 3, parse_timespan('3 h'))
    self.assertEqual(60 * 60 * 3, parse_timespan('3 hours'))
    self.assertEqual(60 * 60 * 24 * 4, parse_timespan('4d'))
    self.assertEqual(60 * 60 * 24 * 4, parse_timespan('4 days'))
    self.assertEqual(60 * 60 * 24 * 7 * 5, parse_timespan('5 w'))
    self.assertEqual(60 * 60 * 24 * 7 * 5, parse_timespan('5 weeks'))
    with self.assertRaises(InvalidTimespan):
        parse_timespan('1z')