import copy
import testtools
from testtools.matchers import HasLength
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.deploy_template
def test_deploy_templates_list_marker(self):
    self.api = utils.FakeAPI(fake_responses_pagination)
    self.mgr = ironicclient.v1.deploy_template.DeployTemplateManager(self.api)
    deploy_templates = self.mgr.list(marker=DEPLOY_TEMPLATE['uuid'])
    expect = [('GET', '/v1/deploy_templates/?marker=%s' % DEPLOY_TEMPLATE['uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertThat(deploy_templates, HasLength(1))