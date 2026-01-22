from designateclient import exceptions
from designateclient.tests import base
def test_get_error_message_with_unknown_response(self):
    expected_msg = 'invalid_object'
    self.response_dict['message'] = expected_msg
    self.response_dict['unknown'] = 'fake'
    remote_err = exceptions.RemoteError(**self.response_dict)
    self.assertEqual(expected_msg, remote_err.message)