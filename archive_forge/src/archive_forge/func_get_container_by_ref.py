from barbicanclient import exceptions
from barbicanclient.v1 import client as barbican_client
from barbicanclient.v1 import containers
from oslo_config import cfg
from oslo_log import log as logging
from heat.common import exception
from heat.engine.clients import client_plugin
from heat.engine import constraints
def get_container_by_ref(self, container_ref):
    try:
        return self.client().containers.get(container_ref)
    except Exception as ex:
        if self.is_not_found(ex):
            raise exception.EntityNotFound(entity='Container', name=container_ref)
        raise