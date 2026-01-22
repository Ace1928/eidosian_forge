import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
def test_pty(self) -> None:
    with sinks.TTYFixture() as out:
        ap = autopage.AutoPager(out.stream)
        self.assertTrue(ap.to_terminal())