from tempfile import NamedTemporaryFile
from textwrap import dedent, indent
from unittest import TestCase, skipUnless
def test_eventual_result_stash_signature(self) -> None:
    """
        EventualResult's stash() method takes no arguments and returns the same type
        retrieve_result's one result_id parameter takes.
        """
    _assert_mypy(True, dedent('                from crochet import EventualResult, retrieve_result\n                def foo(er: EventualResult[object]) -> None:\n                    retrieve_result(er.stash())\n                    retrieve_result(result_id=er.stash())\n                '))