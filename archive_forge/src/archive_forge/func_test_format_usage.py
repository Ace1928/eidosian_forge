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
def test_format_usage(self):
    """Test :func:`humanfriendly.usage.format_usage()`."""
    usage_text = 'Just one --option'
    formatted_text = format_usage(usage_text)
    assert len(formatted_text) > len(usage_text)
    assert formatted_text.startswith('Just one ')
    usage_text = 'Usage: humanfriendly [OPTIONS]'
    formatted_text = format_usage(usage_text)
    assert len(formatted_text) > len(usage_text)
    assert usage_text in formatted_text
    assert not formatted_text.startswith(usage_text)
    usage_text = '--valid-option=VALID_METAVAR\nVALID_METAVAR is bogus\nINVALID_METAVAR should not be highlighted\n'
    formatted_text = format_usage(usage_text)
    formatted_lines = formatted_text.splitlines()
    assert ANSI_CSI in formatted_lines[1]
    assert ANSI_CSI not in formatted_lines[2]