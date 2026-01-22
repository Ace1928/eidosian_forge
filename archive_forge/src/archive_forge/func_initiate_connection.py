import base64
from enum import Enum, IntEnum
from hyperframe.exceptions import InvalidPaddingError
from hyperframe.frame import (
from hpack.hpack import Encoder, Decoder
from hpack.exceptions import HPACKError, OversizedHeaderListError
from .config import H2Configuration
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .frame_buffer import FrameBuffer
from .settings import Settings, SettingCodes
from .stream import H2Stream, StreamClosedBy
from .utilities import SizeLimitDict, guard_increment_window
from .windows import WindowManager
def initiate_connection(self):
    """
        Provides any data that needs to be sent at the start of the connection.
        Must be called for both clients and servers.
        """
    self.config.logger.debug('Initializing connection')
    self.state_machine.process_input(ConnectionInputs.SEND_SETTINGS)
    if self.config.client_side:
        preamble = b'PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n'
    else:
        preamble = b''
    f = SettingsFrame(0)
    for setting, value in self.local_settings.items():
        f.settings[setting] = value
    self.config.logger.debug('Send Settings frame: %s', self.local_settings)
    self._data_to_send += preamble + f.serialize()