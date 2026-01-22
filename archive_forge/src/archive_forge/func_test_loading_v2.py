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
def test_loading_v2(self):
    section = uuid.uuid4().hex
    auth_url = uuid.uuid4().hex
    username = uuid.uuid4().hex
    password = uuid.uuid4().hex
    trust_id = uuid.uuid4().hex
    tenant_id = uuid.uuid4().hex
    self.conf_fixture.config(auth_section=section, group=self.GROUP)
    loading.register_auth_conf_options(self.conf_fixture.conf, group=self.GROUP)
    opts = loading.get_auth_plugin_conf_options(v2.Password())
    self.conf_fixture.register_opts(opts, group=section)
    self.conf_fixture.config(auth_type=self.V2PASS, auth_url=auth_url, username=username, password=password, trust_id=trust_id, tenant_id=tenant_id, group=section)
    a = loading.load_auth_from_conf_options(self.conf_fixture.conf, self.GROUP)
    self.assertEqual(auth_url, a.auth_url)
    self.assertEqual(username, a.username)
    self.assertEqual(password, a.password)
    self.assertEqual(trust_id, a.trust_id)
    self.assertEqual(tenant_id, a.tenant_id)