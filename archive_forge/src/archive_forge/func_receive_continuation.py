from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def receive_continuation(self):
    """
        A naked CONTINUATION frame has been received. This is always an error,
        but the type of error it is depends on the state of the stream and must
        transition the state of the stream, so we need to handle it.
        """
    self.config.logger.debug('Receive Continuation frame on %r', self)
    self.state_machine.process_input(StreamInputs.RECV_CONTINUATION)
    assert False, 'Should not be reachable'