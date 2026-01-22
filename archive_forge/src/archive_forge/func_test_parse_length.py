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
def test_parse_length(self):
    """Test :func:`humanfriendly.parse_length()`."""
    self.assertEqual(0, parse_length('0m'))
    self.assertEqual(42, parse_length('42'))
    self.assertEqual(1.5, parse_length('1.5'))
    self.assertEqual(42, parse_length('42m'))
    self.assertEqual(1000, parse_length('1km'))
    self.assertEqual(0.153, parse_length('15.3 cm'))
    self.assertEqual(0.01, parse_length('1cm'))
    self.assertEqual(0.001, parse_length('1mm'))
    self.assertEqual(1e-09, parse_length('1nm'))
    with self.assertRaises(InvalidLength):
        parse_length('1z')
    with self.assertRaises(InvalidLength):
        parse_length('a')