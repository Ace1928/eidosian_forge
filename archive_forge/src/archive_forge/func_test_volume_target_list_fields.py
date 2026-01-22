import copy
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
def test_volume_target_list_fields(self):
    volume_targets = self.mgr.list(fields=['uuid', 'boot_index'])
    expect = [('GET', '/v1/volume/targets/?fields=uuid,boot_index', {}, None)]
    expect_targets = [TARGET1]
    self._validate_list(expect, expect_targets, volume_targets)