import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
def test_base_exception(self) -> None:

    class MyBaseException(BaseException):
        pass

    def run() -> None:
        with self.ap:
            raise MyBaseException
    self.assertRaises(MyBaseException, run)
    self.assertEqual(1, self.ap.exit_code())