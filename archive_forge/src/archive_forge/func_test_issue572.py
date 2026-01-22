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
def test_issue572(self):
    self.set_input_line('SpammitySpam(')
    self.assertTrue(self.repl.get_args())
    self.set_input_line('WonderfulSpam(')
    self.assertTrue(self.repl.get_args())