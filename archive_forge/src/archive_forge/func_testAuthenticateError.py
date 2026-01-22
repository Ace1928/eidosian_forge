import sys
import mock
from pyu2f import errors
from pyu2f import model
from pyu2f import u2f
def testAuthenticateError(self):
    mock_sk = mock.MagicMock()
    mock_sk.CmdAuthenticate.side_effect = errors.ApduError(255, 255)
    mock_sk.CmdVersion.return_value = b'U2F_V2'
    u2f_api = u2f.U2FInterface(mock_sk)
    with self.assertRaises(errors.U2FError) as cm:
        u2f_api.Authenticate('testapp', b'ABCD', [model.RegisteredKey('khA')])
    self.assertEquals(cm.exception.code, errors.U2FError.BAD_REQUEST)
    self.assertEquals(cm.exception.cause.sw1, 255)
    self.assertEquals(cm.exception.cause.sw2, 255)
    self.assertEquals(mock_sk.CmdAuthenticate.call_count, 1)
    self.assertEquals(mock_sk.CmdWink.call_count, 0)