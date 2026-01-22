from io import StringIO
from typing import Optional, Union
import twisted.trial.unittest
from ...runner import _exit
from .._exit import ExitStatus, exit
def test_exitMessageZero(self) -> None:
    """
        L{exit} given a status code of zero (C{0}) writes the given message to
        standard output.
        """
    out = StringIO()
    self.patch(_exit, 'stdout', out)
    message = 'Hello, world.'
    exit(0, message)
    self.assertEqual(out.getvalue(), message + '\n')