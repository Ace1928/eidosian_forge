from fixtures import Fixture
import signal
from typing import Union
from ._deferreddebug import DebugTwisted
from twisted.internet import defer
from twisted.internet.interfaces import IReactorThreads
from twisted.python.failure import Failure
from twisted.python.util import mergeFunctionMetadata
def clear_junk(self):
    """Clear out our recorded junk.

        :return: Whatever junk was there before.
        """
    junk = self._junk
    self._junk = []
    return junk