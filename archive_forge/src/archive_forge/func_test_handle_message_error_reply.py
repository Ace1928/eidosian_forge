from pyviz_comms import Comm, JupyterComm
from holoviews.element.comparison import ComparisonTestCase
def test_handle_message_error_reply(self):

    def raise_error(msg=None, metadata=None):
        raise Exception('Test')

    def assert_error(msg=None, metadata=None):
        self.assertEqual(metadata['msg_type'], 'Error')
        self.assertTrue(metadata['traceback'].endswith('Exception: Test'))
    comm = Comm(id='Test', on_msg=raise_error)
    comm.send = assert_error
    comm._handle_msg({})