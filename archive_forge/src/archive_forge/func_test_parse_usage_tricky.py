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
def test_parse_usage_tricky(self):
    """Test :func:`humanfriendly.usage.parse_usage()` (a tricky case)."""
    introduction, options = self.preprocess_parse_result("\n            Usage: my-fancy-app [OPTIONS]\n\n            Here's the introduction to my-fancy-app. Some of the lines in the\n            introduction start with a command line option just to confuse the\n            parsing algorithm :-)\n\n            For example\n            --an-awesome-option\n            is still part of the introduction.\n\n            Supported options:\n\n              -a, --an-awesome-option\n\n                Explanation why this is an awesome option.\n\n              -b, --a-boring-option\n\n                Explanation why this is a boring option.\n        ")
    assert 'Usage: my-fancy-app [OPTIONS]' in introduction
    assert any(('still part of the introduction' in p for p in introduction))
    assert 'Supported options:' in introduction
    assert '-a, --an-awesome-option' in options
    assert 'Explanation why this is an awesome option.' in options
    assert '-b, --a-boring-option' in options
    assert 'Explanation why this is a boring option.' in options