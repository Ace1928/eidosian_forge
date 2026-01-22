from io import StringIO
from typing import Optional, Union
import twisted.trial.unittest
from ...runner import _exit
from .._exit import ExitStatus, exit
def test_exitStatusInt(self) -> None:
    """
        L{exit} given an L{int} status code will pass it to L{sys.exit}.
        """
    status = 1234
    exit(status)
    self.assertEqual(self.exit.arg, status)