from oslo_log import log as logging
from zaqarclient.queues.v2 import client as zaqarclient
from zaqarclient.transport import errors as zaqar_errors
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine import constraints
def create_for_tenant(self, tenant_id, token):
    con = self.context
    if token is None:
        LOG.error('Zaqar connection failed, no auth_token!')
        return None
    opts = {'os_auth_token': token, 'os_auth_url': con.auth_url, 'os_project_id': tenant_id, 'os_service_type': self.MESSAGING}
    auth_opts = {'backend': 'keystone', 'options': opts}
    conf = {'auth_opts': auth_opts}
    endpoint = self.url_for(service_type=self.MESSAGING)
    return zaqarclient.Client(url=endpoint, conf=conf, version=2)