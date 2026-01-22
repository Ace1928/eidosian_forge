import copy
import testtools
from testtools.matchers import HasLength
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.deploy_template
def test_deploy_templates_show(self):
    deploy_template = self.mgr.get(DEPLOY_TEMPLATE['uuid'])
    expect = [('GET', '/v1/deploy_templates/%s' % DEPLOY_TEMPLATE['uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(DEPLOY_TEMPLATE['uuid'], deploy_template.uuid)
    self.assertEqual(DEPLOY_TEMPLATE['name'], deploy_template.name)
    self.assertEqual(DEPLOY_TEMPLATE['steps'], deploy_template.steps)
    self.assertEqual(DEPLOY_TEMPLATE['extra'], deploy_template.extra)