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
def test_clean_output(self):
    """Test :func:`humanfriendly.terminal.clean_terminal_output()`."""
    assert clean_terminal_output('foo') == ['foo']
    assert clean_terminal_output('foo\nbar') == ['foo', 'bar']
    assert clean_terminal_output('foo\rbar\nbaz') == ['bar', 'baz']
    assert clean_terminal_output('aaa\rab') == ['aba']
    assert clean_terminal_output('aaa\x08\x08b') == ['aba']
    assert clean_terminal_output('foo\nbar\nbaz\n\n\n') == ['foo', 'bar', 'baz']