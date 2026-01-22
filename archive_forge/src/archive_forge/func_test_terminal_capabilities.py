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
def test_terminal_capabilities(self):
    """Test the functions that check for terminal capabilities."""
    from capturer import CaptureOutput
    for test_stream in (connected_to_terminal, terminal_supports_colors):
        for stream in (sys.stdout, sys.stderr):
            with CaptureOutput():
                assert test_stream(stream)
        with open(os.devnull) as handle:
            assert not test_stream(handle)
        assert not test_stream(object())