import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
def test_broken_pipe(self) -> None:
    with self.ap:
        raise BrokenPipeError
    self.assertEqual(141, self.ap.exit_code())