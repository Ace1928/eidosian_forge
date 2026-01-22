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
def test_retry_return(self):
    """Test :func:`~humanfriendly.testing.retry()` based on return values."""

    def success_helper():
        if not hasattr(success_helper, 'was_called'):
            setattr(success_helper, 'was_called', True)
            return False
        else:
            return 42
    assert retry(success_helper) == 42
    with self.assertRaises(CallableTimedOut):
        retry(lambda: False, timeout=1)