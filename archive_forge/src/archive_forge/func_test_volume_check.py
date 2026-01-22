import copy
from unittest import mock
from cinderclient import exceptions as cinder_exp
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import cinder
from heat.engine.clients.os import nova
from heat.engine.resources.aws.ec2 import instance
from heat.engine.resources.aws.ec2 import volume as aws_vol
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests.openstack.cinder import test_volume_utils as vt_base
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_volume_check(self):
    stack = utils.parse_stack(self.t, stack_name='volume_check')
    res = stack['DataVolume']
    res.state_set(res.CREATE, res.COMPLETE)
    fake_volume = vt_base.FakeVolume('available')
    cinder = mock.Mock()
    cinder.volumes.get.return_value = fake_volume
    self.patchobject(res, 'client', return_value=cinder)
    scheduler.TaskRunner(res.check)()
    self.assertEqual((res.CHECK, res.COMPLETE), res.state)
    fake_volume = vt_base.FakeVolume('in-use')
    res.client().volumes.get.return_value = fake_volume
    scheduler.TaskRunner(res.check)()
    self.assertEqual((res.CHECK, res.COMPLETE), res.state)