from tempfile import NamedTemporaryFile
from textwrap import dedent, indent
from unittest import TestCase, skipUnless
def test_version_string(self) -> None:
    """
        __version__ is a string.
        """
    _assert_mypy(True, dedent('                import crochet\n                x: str = crochet.__version__\n                '))