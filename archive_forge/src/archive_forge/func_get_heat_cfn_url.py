from oslo_config import cfg
from heatclient import client as hc
from heatclient import exc
from heat.engine.clients import client_plugin
def get_heat_cfn_url(self):
    endpoint_type = self._get_client_option(CLIENT_NAME, 'endpoint_type')
    heat_cfn_url = self.url_for(service_type=self.CLOUDFORMATION, endpoint_type=endpoint_type)
    return heat_cfn_url