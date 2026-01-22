import functools
import uuid
import flask
from oslo_log import log
from pycadf import cadftaxonomy as taxonomy
from urllib import parse
from keystone.auth import plugins as auth_plugins
from keystone.auth.plugins import base
from keystone.common import provider_api
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.federation import utils
from keystone.i18n import _
from keystone import notifications
def validate_shadow_mapping(shadow_projects, existing_roles, user_domain_id, idp_id):
    for shadow_project in shadow_projects:
        for shadow_role in shadow_project['roles']:
            if shadow_role['name'] not in existing_roles:
                LOG.error('Role %s was specified in the mapping but does not exist. All roles specified in a mapping must exist before assignment.', shadow_role['name'])
                raise exception.RoleNotFound(shadow_role['name'])
            role = existing_roles[shadow_role['name']]
            if role['domain_id'] is not None and role['domain_id'] != user_domain_id:
                LOG.error('Role %(role)s is a domain-specific role and cannot be assigned within %(domain)s.', {'role': shadow_role['name'], 'domain': user_domain_id})
                raise exception.DomainSpecificRoleNotWithinIdPDomain(role_name=shadow_role['name'], identity_provider=idp_id)