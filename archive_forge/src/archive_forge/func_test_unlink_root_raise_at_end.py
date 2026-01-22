from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick import privileged
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
@mock.patch.object(priv_rootwrap.unlink_root.privsep_entrypoint, 'client_mode', False)
@mock.patch('os.unlink', side_effect=IOError)
def test_unlink_root_raise_at_end(self, unlink_mock):
    links = ['/dev/disk/by-id/link1', '/dev/disk/by-id/link2']
    self.assertRaises(exception.ExceptionChainer, priv_rootwrap.unlink_root, *links, raise_at_end=True)
    unlink_mock.assert_has_calls([mock.call(links[0]), mock.call(links[1])])