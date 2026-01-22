import collections
import copy
import json
from unittest import mock
from cinderclient import exceptions as cinder_exp
from novaclient import exceptions as nova_exp
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import cinder
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine.resources.openstack.cinder import volume as c_vol
from heat.engine.resources import scheduler_hints as sh
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.objects import resource_data as resource_data_object
from heat.tests.openstack.cinder import test_volume_utils as vt_base
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_cinder_fn_getatt(self):
    self.stack_name = 'test_cvolume_fngetatt_stack'
    fv = vt_base.FakeVolume('available', availability_zone='zone1', size=1, snapshot_id='snap-123', name='name', description='desc', volume_type='lvm', metadata={'key': 'value'}, source_volid=None, bootable=False, created_at='2013-02-25T02:40:21.000000', encrypted=False, attachments=[])
    self._mock_create_volume(vt_base.FakeVolume('creating'), self.stack_name, extra_get_mocks=[fv for i in range(20)])
    stack = utils.parse_stack(self.t, stack_name=self.stack_name)
    rsrc = self.create_volume(self.t, stack, 'volume')
    self.assertEqual(u'zone1', rsrc.FnGetAtt('availability_zone'))
    self.assertEqual(u'1', rsrc.FnGetAtt('size'))
    self.assertEqual(u'snap-123', rsrc.FnGetAtt('snapshot_id'))
    self.assertEqual(u'name', rsrc.FnGetAtt('display_name'))
    self.assertEqual(u'desc', rsrc.FnGetAtt('display_description'))
    self.assertEqual(u'lvm', rsrc.FnGetAtt('volume_type'))
    self.assertEqual(json.dumps({'key': 'value'}), rsrc.FnGetAtt('metadata'))
    self.assertEqual({'key': 'value'}, rsrc.FnGetAtt('metadata_values'))
    self.assertEqual(u'None', rsrc.FnGetAtt('source_volid'))
    self.assertEqual(u'available', rsrc.FnGetAtt('status'))
    self.assertEqual(u'2013-02-25T02:40:21.000000', rsrc.FnGetAtt('created_at'))
    self.assertEqual(u'False', rsrc.FnGetAtt('bootable'))
    self.assertEqual(u'False', rsrc.FnGetAtt('encrypted'))
    self.assertEqual(u'[]', rsrc.FnGetAtt('attachments'))
    self.assertEqual([], rsrc.FnGetAtt('attachments_list'))
    error = self.assertRaises(exception.InvalidTemplateAttribute, rsrc.FnGetAtt, 'unknown')
    self.assertEqual('The Referenced Attribute (volume unknown) is incorrect.', str(error))
    self.cinder_fc.volumes.get.assert_called_with('vol-123')