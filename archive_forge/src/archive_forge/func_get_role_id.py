import re
from keystoneauth1 import exceptions as ks_exceptions
from heat.common import exception
from heat.engine.clients import client_plugin
from heat.engine.clients.os.keystone import heat_keystoneclient as hkc
def get_role_id(self, role, domain=None):
    if role is None:
        return None
    if not domain:
        role, domain = self.parse_entity_with_domain(role, 'KeystoneRole')
    try:
        role_obj = self.client().client.roles.get(role)
        return role_obj.id
    except ks_exceptions.NotFound:
        role_list = self.client().client.roles.list(name=role, domain=domain)
        for role_obj in role_list:
            if role_obj.name == role:
                return role_obj.id
    raise exception.EntityNotFound(entity='KeystoneRole', name=role)