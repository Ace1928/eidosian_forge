from unittest import mock
from os_brick import exception
from os_brick.initiator import linuxrbd
from os_brick.tests import base
from os_brick import utils
@mock.patch.object(MockRados.Rados, 'connect', side_effect=MockRados.Error)
def test_with_client_error(self, _):
    linuxrbd.rados = MockRados
    linuxrbd.rados.Error = MockRados.Error

    def test():
        with linuxrbd.RBDClient('test_user', 'test_pool'):
            pass
    self.assertRaises(exception.BrickException, test)