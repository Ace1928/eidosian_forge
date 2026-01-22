from __future__ import annotations
import collections.abc as c
import datetime
import typing as t
from .util import (
from .util_common import (
from .metadata import (
from .config import (
from . import junit_xml
def write_junit(self, args: TestConfig) -> None:
    """Write results to a junit XML file."""
    title = self.format_title()
    output = self.format_block()
    test_case = junit_xml.TestCase(classname=self.command, name=self.name, failures=[junit_xml.TestFailure(message=title, output=output)])
    self.save_junit(args, test_case)