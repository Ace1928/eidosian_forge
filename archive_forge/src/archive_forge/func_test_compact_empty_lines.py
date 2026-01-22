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
def test_compact_empty_lines(self):
    """Test :func:`humanfriendly.text.compact_empty_lines()`."""
    assert compact_empty_lines('foo') == 'foo'
    assert compact_empty_lines('\tfoo') == '\tfoo'
    assert compact_empty_lines('foo\nbar') == 'foo\nbar'
    assert compact_empty_lines('foo\n\nbar') == 'foo\n\nbar'
    assert compact_empty_lines('foo\n\n\nbar') == 'foo\n\nbar'
    assert compact_empty_lines('foo\n\n\n\nbar') == 'foo\n\nbar'
    assert compact_empty_lines('foo\n\n\n\n\nbar') == 'foo\n\nbar'