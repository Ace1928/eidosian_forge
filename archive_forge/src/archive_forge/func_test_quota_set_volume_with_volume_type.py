import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.common import quota
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
def test_quota_set_volume_with_volume_type(self):
    arglist = ['--gigabytes', str(volume_fakes.QUOTA['gigabytes']), '--snapshots', str(volume_fakes.QUOTA['snapshots']), '--volumes', str(volume_fakes.QUOTA['volumes']), '--backups', str(volume_fakes.QUOTA['backups']), '--backup-gigabytes', str(volume_fakes.QUOTA['backup_gigabytes']), '--per-volume-gigabytes', str(volume_fakes.QUOTA['per_volume_gigabytes']), '--volume-type', 'volume_type_backend', self.projects[0].name]
    verifylist = [('gigabytes', volume_fakes.QUOTA['gigabytes']), ('snapshots', volume_fakes.QUOTA['snapshots']), ('volumes', volume_fakes.QUOTA['volumes']), ('backups', volume_fakes.QUOTA['backups']), ('backup_gigabytes', volume_fakes.QUOTA['backup_gigabytes']), ('per_volume_gigabytes', volume_fakes.QUOTA['per_volume_gigabytes']), ('volume_type', 'volume_type_backend'), ('project', self.projects[0].name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'gigabytes_volume_type_backend': volume_fakes.QUOTA['gigabytes'], 'snapshots_volume_type_backend': volume_fakes.QUOTA['snapshots'], 'volumes_volume_type_backend': volume_fakes.QUOTA['volumes'], 'backups': volume_fakes.QUOTA['backups'], 'backup_gigabytes': volume_fakes.QUOTA['backup_gigabytes'], 'per_volume_gigabytes': volume_fakes.QUOTA['per_volume_gigabytes']}
    self.volume_quotas_mock.update.assert_called_once_with(self.projects[0].id, **kwargs)
    self.assertIsNone(result)