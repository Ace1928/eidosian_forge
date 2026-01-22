import os
import tempfile
from unittest import mock
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.remotefs import remotefs
from os_brick.tests import base
@mock.patch.object(remotefs.RemoteFsClient, '_read_mounts', return_value=[])
def test_vzstorage_with_auth(self, mock_read_mounts):
    client = remotefs.VZStorageRemoteFSClient('vzstorage', root_helper='true', vzstorage_mount_point_base='/mnt')
    cluster_name = 'qwe'
    password = '123456'
    share = '%s:%s' % (cluster_name, password)
    mount_point = client.get_mount_point(share)
    client.mount(share)
    calls = [mock.call('mkdir', '-p', mount_point, check_exit_code=0), mock.call('pstorage', '-c', cluster_name, 'auth-node', '-P', process_input=password, root_helper='true', run_as_root=True), mock.call('pstorage-mount', '-c', cluster_name, mount_point, root_helper='true', check_exit_code=0, run_as_root=True)]
    self.mock_execute.assert_has_calls(calls)