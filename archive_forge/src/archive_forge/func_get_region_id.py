import re
from keystoneauth1 import exceptions as ks_exceptions
from heat.common import exception
from heat.engine.clients import client_plugin
from heat.engine.clients.os.keystone import heat_keystoneclient as hkc
def get_region_id(self, region):
    try:
        region_obj = self.client().client.regions.get(region)
        return region_obj.id
    except ks_exceptions.NotFound:
        raise exception.EntityNotFound(entity='KeystoneRegion', name=region)