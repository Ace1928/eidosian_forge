import re
from keystoneauth1 import exceptions as ks_exceptions
from heat.common import exception
from heat.engine.clients import client_plugin
from heat.engine.clients.os.keystone import heat_keystoneclient as hkc
def get_service_id(self, service):
    if service is None:
        return None
    try:
        service_obj = self.client().client.services.get(service)
        return service_obj.id
    except ks_exceptions.NotFound:
        service_list = self.client().client.services.list(name=service)
        if len(service_list) == 1:
            return service_list[0].id
        elif len(service_list) > 1:
            raise exception.KeystoneServiceNameConflict(service=service)
        else:
            raise exception.EntityNotFound(entity='KeystoneService', name=service)