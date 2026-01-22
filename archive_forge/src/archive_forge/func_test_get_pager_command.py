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
def test_get_pager_command(self):
    """Test :func:`humanfriendly.terminal.get_pager_command()`."""
    assert '--RAW-CONTROL-CHARS' not in get_pager_command('Usage message')
    assert '--RAW-CONTROL-CHARS' in get_pager_command(ansi_wrap('Usage message', bold=True))
    options_specific_to_less = ['--no-init', '--quit-if-one-screen']
    for pager in ('cat', 'less'):
        original_pager = os.environ.get('PAGER', None)
        try:
            os.environ['PAGER'] = pager
            command_line = get_pager_command()
            if pager == 'less':
                assert all((opt in command_line for opt in options_specific_to_less))
            else:
                assert not any((opt in command_line for opt in options_specific_to_less))
        finally:
            if original_pager is not None:
                os.environ['PAGER'] = original_pager
            else:
                os.environ.pop('PAGER')