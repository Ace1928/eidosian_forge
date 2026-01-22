import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
def test_errors_bogus_string(self) -> None:
    self.assertRaises(ValueError, autopage.AutoPager, self.stream, errors='panic')