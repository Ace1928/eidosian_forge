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
def test_sphinx_customizations(self):
    """Test the :mod:`humanfriendly.sphinx` module."""

    class FakeApp(object):

        def __init__(self):
            self.callbacks = {}
            self.roles = {}

        def __documented_special_method__(self):
            """Documented unofficial special method."""
            pass

        def __undocumented_special_method__(self):
            pass

        def add_role(self, name, callback):
            self.roles[name] = callback

        def connect(self, event, callback):
            self.callbacks.setdefault(event, []).append(callback)

        def bogus_usage(self):
            """Usage: This is not supposed to be reformatted!"""
            pass
    fake_app = FakeApp()
    setup(fake_app)
    assert man_role == fake_app.roles['man']
    assert pypi_role == fake_app.roles['pypi']
    assert deprecation_note_callback in fake_app.callbacks['autodoc-process-docstring']
    assert special_methods_callback in fake_app.callbacks['autodoc-skip-member']
    assert usage_message_callback in fake_app.callbacks['autodoc-process-docstring']
    assert special_methods_callback(app=None, what=None, name=None, obj=FakeApp.__documented_special_method__, skip=True, options=None) is False
    assert special_methods_callback(app=None, what=None, name=None, obj=FakeApp.__undocumented_special_method__, skip=True, options=None) is True
    from humanfriendly import cli, sphinx
    assert self.docstring_is_reformatted(cli)
    assert not self.docstring_is_reformatted(sphinx)
    assert not self.docstring_is_reformatted(fake_app.bogus_usage)