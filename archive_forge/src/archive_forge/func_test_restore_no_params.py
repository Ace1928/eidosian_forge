from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v2 import backup
from openstack import exceptions
from openstack.tests.unit import base
def test_restore_no_params(self):
    sot = backup.Backup(**BACKUP)
    self.assertRaises(exceptions.SDKException, sot.restore, self.sess)