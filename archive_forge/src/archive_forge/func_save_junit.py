from __future__ import annotations
import collections.abc as c
import datetime
import typing as t
from .util import (
from .util_common import (
from .metadata import (
from .config import (
from . import junit_xml
def save_junit(self, args: TestConfig, test_case: junit_xml.TestCase) -> None:
    """Save the given test case results to disk as JUnit XML."""
    suites = junit_xml.TestSuites(suites=[junit_xml.TestSuite(name='ansible-test', cases=[test_case], timestamp=datetime.datetime.now(tz=datetime.timezone.utc))])
    report = suites.to_pretty_xml()
    if args.explain:
        return
    write_text_test_results(ResultType.JUNIT, self.create_result_name('.xml'), report)