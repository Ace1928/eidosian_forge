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
def test_prompt_for_choice(self):
    """Test :func:`humanfriendly.prompts.prompt_for_choice()`."""
    with self.assertRaises(ValueError):
        prompt_for_choice([])
    with open(os.devnull) as handle:
        with PatchedAttribute(sys, 'stdin', handle):
            only_option = 'only one option (shortcut)'
            assert prompt_for_choice([only_option]) == only_option
    with PatchedAttribute(prompts, 'interactive_prompt', lambda p: 'foo'):
        assert prompt_for_choice(['foo', 'bar']) == 'foo'
    with PatchedAttribute(prompts, 'interactive_prompt', lambda p: 'f'):
        assert prompt_for_choice(['foo', 'bar']) == 'foo'
    with PatchedAttribute(prompts, 'interactive_prompt', lambda p: '2'):
        assert prompt_for_choice(['foo', 'bar']) == 'bar'
    with PatchedAttribute(prompts, 'interactive_prompt', lambda p: ''):
        assert prompt_for_choice(['foo', 'bar'], default='bar') == 'bar'
    replies = ['', 'q', 'z']
    with PatchedAttribute(prompts, 'interactive_prompt', lambda p: replies.pop(0)):
        assert prompt_for_choice(['foo', 'bar', 'baz']) == 'baz'
    replies = ['a', 'q']
    with PatchedAttribute(prompts, 'interactive_prompt', lambda p: replies.pop(0)):
        assert prompt_for_choice(['foo', 'bar', 'baz', 'qux']) == 'qux'
    replies = ['42', '2']
    with PatchedAttribute(prompts, 'interactive_prompt', lambda p: replies.pop(0)):
        assert prompt_for_choice(['foo', 'bar', 'baz']) == 'bar'
    with PatchedAttribute(prompts, 'interactive_prompt', lambda p: ''):
        with self.assertRaises(TooManyInvalidReplies):
            prompt_for_choice(['a', 'b', 'c'])