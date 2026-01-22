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
def set_default_for_default_log_levels():
    """Set the default for the default_log_levels option for keystone.

    Keystone uses some packages that other OpenStack services don't use that do
    logging. This will set the default_log_levels default level for those
    packages.

    This function needs to be called before CONF().

    """
    extra_log_level_defaults = ['dogpile=INFO', 'routes=INFO']
    log.register_options(CONF)
    log.set_defaults(default_log_levels=log.get_default_log_levels() + extra_log_level_defaults)