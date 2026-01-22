import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
def test_system_exit(self) -> None:

    def run() -> None:
        with self.ap:
            raise SystemExit(42)
    self.assertRaises(SystemExit, run)
    self.assertEqual(42, self.ap.exit_code())