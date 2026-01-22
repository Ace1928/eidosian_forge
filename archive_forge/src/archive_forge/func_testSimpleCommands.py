import sys
import mock
from pyu2f import errors
from pyu2f import hardware
def testSimpleCommands(self):
    mock_transport = mock.MagicMock()
    sk = hardware.SecurityKey(mock_transport)
    sk.CmdBlink(5)
    mock_transport.SendBlink.assert_called_once_with(5)
    sk.CmdWink()
    mock_transport.SendWink.assert_called_once_with()
    sk.CmdPing(bytearray(b'foo'))
    mock_transport.SendPing.assert_called_once_with(bytearray(b'foo'))