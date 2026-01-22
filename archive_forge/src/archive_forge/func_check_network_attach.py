from oslo_config import cfg
import tenacity
from zunclient import client as zun_client
from zunclient import exceptions as zc_exc
from heat.engine.clients import client_plugin
@tenacity.retry(stop=tenacity.stop_after_attempt(cfg.CONF.max_interface_check_attempts), wait=tenacity.wait_exponential(multiplier=0.5, max=12.0), retry=tenacity.retry_if_result(client_plugin.retry_if_result_is_false))
def check_network_attach(self, container_id, port_id):
    if not port_id:
        return True
    interfaces = self.client(version=self.V1_18).containers.network_list(container_id)
    for iface in interfaces:
        if iface.port_id == port_id:
            return True
    return False