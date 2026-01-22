from tempfile import NamedTemporaryFile
from textwrap import dedent, indent
from unittest import TestCase, skipUnless
def test_eventual_result_cancel_signature(self) -> None:
    """
        EventualResult's cancel() method takes no arguments.
        """
    _assert_mypy(True, dedent('                from crochet import EventualResult\n                def foo(er: EventualResult[object]) -> None:\n                    er.cancel()\n                '))