import os
import pytest
import sys
import textwrap
import unittest
from contextlib import contextmanager
from traitlets.config.loader import Config
from IPython import get_ipython
from IPython.core import completer
from IPython.utils.tempdir import TemporaryDirectory, TemporaryWorkingDirectory
from IPython.utils.generics import complete_object
from IPython.testing import decorators as dec
from IPython.core.completer import (
def test_matcher_disabling(self):

    @completion_matcher(identifier='a_matcher')
    def a_matcher(text):
        return ['completion_a']

    @completion_matcher(identifier='b_matcher')
    def b_matcher(text):
        return ['completion_b']

    def _(expected):
        s, matches = c.complete('completion_')
        self.assertEqual(expected, matches)
    with custom_matchers([a_matcher, b_matcher]):
        ip = get_ipython()
        c = ip.Completer
        _(['completion_a', 'completion_b'])
        cfg = Config()
        cfg.IPCompleter.disable_matchers = ['b_matcher']
        c.update_config(cfg)
        _(['completion_a'])
        cfg.IPCompleter.disable_matchers = []
        c.update_config(cfg)