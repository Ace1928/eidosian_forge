from os_brick import exception
from os_brick.tests import base
def test_error_msg_exception_with_kwargs(self):

    class FakeBrickException(exception.BrickException):
        message = 'default message: %(mispelled_code)s'
    exc = FakeBrickException(code=500)
    self.assertEqual(str(exc), 'default message: %(mispelled_code)s')