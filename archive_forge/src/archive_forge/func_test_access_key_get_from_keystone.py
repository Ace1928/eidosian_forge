from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import node_data
from heat.engine.resources.aws.iam import user
from heat.engine.resources.openstack.heat import access_policy as ap
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.objects import resource_data as resource_data_object
from heat.tests import common
from heat.tests import utils
def test_access_key_get_from_keystone(self):
    self.patchobject(user.AccessKey, 'keystone', return_value=self.fc)
    t = template_format.parse(user_accesskey_template)
    stack = utils.parse_stack(t)
    self.create_user(t, stack, 'CfnUser')
    rsrc = self.create_access_key(t, stack, 'HostKeys')
    resource_data_object.ResourceData.delete(rsrc, 'credential_id')
    resource_data_object.ResourceData.delete(rsrc, 'secret_key')
    self.assertRaises(exception.NotFound, resource_data_object.ResourceData.get_all, rsrc)
    rsrc._secret = None
    rsrc._data = None
    self.assertEqual(self.fc.secret, rsrc.FnGetAtt('SecretAccessKey'))
    scheduler.TaskRunner(rsrc.delete)()
    self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)