import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
def test_defaults_cmd_as_int(self) -> None:
    self.assertRaises(TypeError, autopage.AutoPager, pager_command=42)