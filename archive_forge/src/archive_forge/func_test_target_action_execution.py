import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_target_action_execution(self):
    command = '--debug --os-target-tenant-name={tenantname} --os-target-username={username} --os-target-password="{password}" --os-target-auth-url="{auth_url}" --target_insecure run-action std.noop'.format(tenantname=self.clients.tenant_name, username=self.clients.username, password=self.clients.password, auth_url=self.clients.uri)
    self.mistral_alt_user(cmd=command)