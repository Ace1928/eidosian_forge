from unittest import mock
import uuid
from oslo_config import cfg
from oslo_config import fixture as config
import stevedore
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.loading._plugins.identity import v2
from keystoneauth1.loading._plugins.identity import v3
from keystoneauth1.tests.unit.loading import utils
def test_loading_v3(self):
    section = uuid.uuid4().hex
    auth_url = (uuid.uuid4().hex,)
    token = uuid.uuid4().hex
    trust_id = uuid.uuid4().hex
    project_id = uuid.uuid4().hex
    project_domain_name = uuid.uuid4().hex
    self.conf_fixture.config(auth_section=section, group=self.GROUP)
    loading.register_auth_conf_options(self.conf_fixture.conf, group=self.GROUP)
    opts = loading.get_auth_plugin_conf_options(v3.Token())
    self.conf_fixture.register_opts(opts, group=section)
    self.conf_fixture.config(auth_type=self.V3TOKEN, auth_url=auth_url, token=token, trust_id=trust_id, project_id=project_id, project_domain_name=project_domain_name, group=section)
    a = loading.load_auth_from_conf_options(self.conf_fixture.conf, self.GROUP)
    self.assertEqual(token, a.auth_methods[0].token)
    self.assertEqual(trust_id, a.trust_id)
    self.assertEqual(project_id, a.project_id)
    self.assertEqual(project_domain_name, a.project_domain_name)