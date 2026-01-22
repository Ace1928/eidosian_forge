from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def response_sent(self, previous_state):
    """
        Fires when something that should be a response is sent. This 'response'
        may actually be trailers.
        """
    if not self.headers_sent:
        if self.client is True or self.client is None:
            raise ProtocolError('Client cannot send responses.')
        self.headers_sent = True
        event = _ResponseSent()
    else:
        assert not self.trailers_sent
        self.trailers_sent = True
        event = _TrailersSent()
    return [event]