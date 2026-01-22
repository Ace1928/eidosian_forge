from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def end_stream(self):
    """
        End a stream without sending data.
        """
    self.config.logger.debug('End stream %r', self)
    self.state_machine.process_input(StreamInputs.SEND_END_STREAM)
    df = DataFrame(self.stream_id)
    df.flags.add('END_STREAM')
    return [df]