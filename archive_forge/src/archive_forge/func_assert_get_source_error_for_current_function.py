import collections
import inspect
import socket
import sys
import tempfile
import unittest
from typing import List, Tuple
from itertools import islice
from pathlib import Path
from unittest import mock
from bpython import config, repl, cli, autocomplete
from bpython.line import LinePart
from bpython.test import (
def assert_get_source_error_for_current_function(self, func, msg):
    self.repl.current_func = func
    with self.assertRaises(repl.SourceNotFound):
        self.repl.get_source_of_current_name()
    try:
        self.repl.get_source_of_current_name()
    except repl.SourceNotFound as e:
        self.assertEqual(e.args[0], msg)
    else:
        self.fail('Should have raised SourceNotFound')