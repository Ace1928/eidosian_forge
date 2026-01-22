from tempfile import NamedTemporaryFile
from textwrap import dedent, indent
from unittest import TestCase, skipUnless
def test_run_in_reactor_func_returns_typed_eventual(self) -> None:
    """
        run_in_reactor preserves the decorated function's return type indirectly
        through an EventualResult.
        """
    template = dedent('            from typing import Optional\n            from crochet import EventualResult, run_in_reactor\n\n            @run_in_reactor\n            def foo() -> {return_type}:\n                return {return_value}\n\n            eventual_result: {receiver_type} = foo()\n            final_result: {final_type} = eventual_result.wait(1)\n            ')
    for return_type, return_value, receiver_type, final_type, good in (('int', '1', 'EventualResult[int]', 'int', True), ('int', "'str'", 'EventualResult[int]', 'int', False), ('int', '1', 'EventualResult[str]', 'int', False), ('int', '1', 'EventualResult[str]', 'str', False), ('int', '1', 'int', 'int', False), ('int', '1', 'EventualResult[int]', 'Optional[int]', True), ('Optional[int]', '1', 'EventualResult[Optional[int]]', 'Optional[int]', True), ('Optional[int]', 'None', 'EventualResult[Optional[int]]', 'Optional[int]', True), ('Optional[int]', '1', 'EventualResult[int]', 'Optional[int]', False), ('Optional[int]', '1', 'EventualResult[Optional[int]]', 'int', False)):
        with self.subTest(return_type=return_type, return_value=return_value, receiver_type=receiver_type, final_type=final_type):
            _assert_mypy(good, template.format(return_type=return_type, return_value=return_value, receiver_type=receiver_type, final_type=final_type))