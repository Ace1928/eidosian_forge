import json
from keystoneauth1 import loading as ks_loading
from oslo_log import log as logging
from heat.common import exception
def get_keystone_plugin_loader(auth, keystone_session):
    cred = parse_auth_credential_to_dict(auth)
    auth_plugin = ks_loading.get_plugin_loader(cred.get('auth_type')).load_from_options(**cred.get('auth'))
    validate_auth_plugin(auth_plugin, keystone_session)
    return auth_plugin