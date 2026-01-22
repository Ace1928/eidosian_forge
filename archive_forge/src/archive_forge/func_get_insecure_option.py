from oslo_config import cfg
from heatclient import client as hc
from heatclient import exc
from heat.engine.clients import client_plugin
def get_insecure_option(self):
    return self._get_client_option(CLIENT_NAME, 'insecure')