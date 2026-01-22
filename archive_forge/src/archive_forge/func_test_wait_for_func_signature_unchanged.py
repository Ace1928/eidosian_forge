from tempfile import NamedTemporaryFile
from textwrap import dedent, indent
from unittest import TestCase, skipUnless
def test_wait_for_func_signature_unchanged(self) -> None:
    """
        The @wait_for(timeout) decorator preserves the wrapped function's signature.
        """
    template = dedent('            from typing import Callable\n            from crochet import wait_for\n\n            class Thing:\n                pass\n\n            @wait_for(1)\n            def foo(x: int, y: str, z: float) -> Thing:\n                return Thing()\n\n            re_foo: {result_type} = foo\n            ')
    for result_type, good in (('Callable[[int, str, float], Thing]', True), ('Callable[[int, str, float], object]', True), ('Callable[[int, str, float], int]', False), ('Callable[[int, str, float], EventualResult[Thing]]', False), ('Callable[[int, str, float], None]', False), ('Callable[[int, str], Thing]', False), ('Callable[[int], Thing]', False), ('Callable[[], Thing]', False), ('Callable[[float, int, str], Thing]', False)):
        with self.subTest(result_type=result_type):
            _assert_mypy(good, template.format(result_type=result_type))