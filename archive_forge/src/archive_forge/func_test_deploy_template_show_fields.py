import copy
import testtools
from testtools.matchers import HasLength
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.deploy_template
def test_deploy_template_show_fields(self):
    deploy_template = self.mgr.get(DEPLOY_TEMPLATE['uuid'], fields=['uuid', 'name'])
    expect = [('GET', '/v1/deploy_templates/%s?fields=uuid,name' % DEPLOY_TEMPLATE['uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(DEPLOY_TEMPLATE['uuid'], deploy_template.uuid)
    self.assertEqual(DEPLOY_TEMPLATE['name'], deploy_template.name)