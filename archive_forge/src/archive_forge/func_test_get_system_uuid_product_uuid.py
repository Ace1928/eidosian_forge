import builtins
import errno
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
import os_brick.privileged as privsep_brick
import os_brick.privileged.nvmeof as privsep_nvme
from os_brick.privileged import rootwrap
from os_brick.tests import base
@mock.patch.object(builtins, 'open', new_callable=mock.mock_open)
def test_get_system_uuid_product_uuid(self, mock_open):
    uuid = 'dbc6ba60-36ae-4b96-9310-628832bdfc3d'
    mock_fd = mock_open.return_value.__enter__.return_value
    mock_fd.read.return_value = uuid
    res = privsep_nvme.get_system_uuid()
    self.assertEqual(uuid, res)
    mock_open.assert_called_once_with('/sys/class/dmi/id/product_uuid', 'r')
    mock_fd.read.assert_called_once_with()