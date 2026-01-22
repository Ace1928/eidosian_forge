import logging
from oslo_cache import core as cache
from oslo_config import cfg
from oslo_log import log
import oslo_messaging
from oslo_middleware import cors
from oslo_policy import opts as policy_opts
from osprofiler import opts as profiler
from keystone.conf import application_credential
from keystone.conf import assignment
from keystone.conf import auth
from keystone.conf import catalog
from keystone.conf import credential
from keystone.conf import default
from keystone.conf import domain_config
from keystone.conf import endpoint_filter
from keystone.conf import endpoint_policy
from keystone.conf import federation
from keystone.conf import fernet_receipts
from keystone.conf import fernet_tokens
from keystone.conf import identity
from keystone.conf import identity_mapping
from keystone.conf import jwt_tokens
from keystone.conf import ldap
from keystone.conf import oauth1
from keystone.conf import oauth2
from keystone.conf import policy
from keystone.conf import receipt
from keystone.conf import resource
from keystone.conf import revoke
from keystone.conf import role
from keystone.conf import saml
from keystone.conf import security_compliance
from keystone.conf import shadow_users
from keystone.conf import token
from keystone.conf import tokenless_auth
from keystone.conf import totp
from keystone.conf import trust
from keystone.conf import unified_limit
from keystone.conf import wsgi
def set_external_opts_defaults():
    """Update default configuration options for oslo.middleware."""
    cors.set_defaults(allow_headers=['X-Auth-Token', 'X-Openstack-Request-Id', 'X-Subject-Token', 'X-Project-Id', 'X-Project-Name', 'X-Project-Domain-Id', 'X-Project-Domain-Name', 'X-Domain-Id', 'X-Domain-Name', 'Openstack-Auth-Receipt'], expose_headers=['X-Auth-Token', 'X-Openstack-Request-Id', 'X-Subject-Token', 'Openstack-Auth-Receipt'], allow_methods=['GET', 'PUT', 'POST', 'DELETE', 'PATCH'])
    profiler.set_defaults(CONF, enabled=False, trace_sqlalchemy=False)
    DEFAULT_POLICY_FILE = 'policy.yaml'
    policy_opts.set_defaults(cfg.CONF, DEFAULT_POLICY_FILE)
    opts = cache._opts.list_opts()
    for opt_list in opts:
        if opt_list[0] == 'cache':
            for o in opt_list[1]:
                if o.name == 'enabled':
                    o.default = True