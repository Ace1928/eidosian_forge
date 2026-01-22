from tempfile import NamedTemporaryFile
from textwrap import dedent, indent
from unittest import TestCase, skipUnless
def test_eventual_result_original_failure_signature(self) -> None:
    """
        EventualResult's original_failure() method takes no arguments and returns an
        optional Failure.
        """
    _assert_mypy(True, dedent('                from typing import Optional\n                from twisted.python.failure import Failure\n                from crochet import EventualResult\n                def foo(er: EventualResult[object]) -> Optional[Failure]:\n                    return er.original_failure()\n                '))
    _assert_mypy(False, dedent('                from twisted.python.failure import Failure\n                from crochet import EventualResult\n                def foo(er: EventualResult[object]) -> Failure:\n                    return er.original_failure()\n                '))