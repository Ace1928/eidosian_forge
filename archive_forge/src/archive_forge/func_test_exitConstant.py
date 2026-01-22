from io import StringIO
from typing import Optional, Union
import twisted.trial.unittest
from ...runner import _exit
from .._exit import ExitStatus, exit
def test_exitConstant(self) -> None:
    """
        L{exit} given a L{ValueConstant} status code passes the corresponding
        value to L{sys.exit}.
        """
    status = ExitStatus.EX_CONFIG
    exit(status)
    self.assertEqual(self.exit.arg, status.value)