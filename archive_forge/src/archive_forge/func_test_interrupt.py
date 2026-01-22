import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
def test_interrupt(self) -> None:

    def run() -> None:
        with self.ap:
            raise KeyboardInterrupt
    self.assertRaises(KeyboardInterrupt, run)
    self.assertEqual(130, self.ap.exit_code())