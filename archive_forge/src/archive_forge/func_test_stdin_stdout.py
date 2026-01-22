from __future__ import with_statement
import os
import sys
import textwrap
import unittest
import subprocess
import tempfile
def test_stdin_stdout(self):
    self.assertEqual(self.runTool(data=self.data.encode()), self.expect.splitlines())