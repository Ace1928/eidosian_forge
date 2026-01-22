import sys
import mock
from pyu2f import errors
from pyu2f import hardware
def testAuthenticateTUPRequired(self):
    mock_transport = mock.MagicMock()
    sk = hardware.SecurityKey(mock_transport)
    challenge_param = b'01234567890123456789012345678901'
    app_param = b'01234567890123456789012345678901'
    key_handle = b'\x01\x02\x03\x04'
    mock_transport.SendMsgBytes.return_value = bytearray([105, 133])
    self.assertRaises(errors.TUPRequiredError, sk.CmdAuthenticate, challenge_param, app_param, key_handle)
    self.assertEquals(mock_transport.SendMsgBytes.call_count, 1)