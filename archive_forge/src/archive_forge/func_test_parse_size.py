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
def test_parse_size(self):
    """Test :func:`humanfriendly.parse_size()`."""
    self.assertEqual(0, parse_size('0B'))
    self.assertEqual(42, parse_size('42'))
    self.assertEqual(42, parse_size('42B'))
    self.assertEqual(1000, parse_size('1k'))
    self.assertEqual(1024, parse_size('1k', binary=True))
    self.assertEqual(1000, parse_size('1 KB'))
    self.assertEqual(1000, parse_size('1 kilobyte'))
    self.assertEqual(1024, parse_size('1 kilobyte', binary=True))
    self.assertEqual(1000 ** 2 * 69, parse_size('69 MB'))
    self.assertEqual(1000 ** 3, parse_size('1 GB'))
    self.assertEqual(1000 ** 4, parse_size('1 TB'))
    self.assertEqual(1000 ** 5, parse_size('1 PB'))
    self.assertEqual(1000 ** 6, parse_size('1 EB'))
    self.assertEqual(1000 ** 7, parse_size('1 ZB'))
    self.assertEqual(1000 ** 8, parse_size('1 YB'))
    self.assertEqual(1000 ** 3 * 1.5, parse_size('1.5 GB'))
    self.assertEqual(1024 ** 8 * 1.5, parse_size('1.5 YiB'))
    with self.assertRaises(InvalidSize):
        parse_size('1q')
    with self.assertRaises(InvalidSize):
        parse_size('a')