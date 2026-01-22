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
def test_alias_proxy_sphinx_integration(self):
    """Test that aliases can be injected into generated documentation."""
    module = sys.modules[__name__]
    define_aliases(__name__, concatenate='humanfriendly.text.concatenate')
    lines = module.__doc__.splitlines()
    deprecation_note_callback(app=None, what=None, name=None, obj=module, options=None, lines=lines)
    assert '\n'.join(lines) != module.__doc__