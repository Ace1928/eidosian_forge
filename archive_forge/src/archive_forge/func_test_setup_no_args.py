from tempfile import NamedTemporaryFile
from textwrap import dedent, indent
from unittest import TestCase, skipUnless
def test_setup_no_args(self) -> None:
    """
        setup() and no_setup() take no arguments.
        """
    _assert_mypy(True, dedent('\n                from crochet import setup\n                setup()\n                '))
    _assert_mypy(True, dedent('\n                from crochet import no_setup\n                no_setup()\n                '))