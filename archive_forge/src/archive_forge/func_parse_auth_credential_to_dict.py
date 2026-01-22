import json
from keystoneauth1 import loading as ks_loading
from oslo_log import log as logging
from heat.common import exception
def parse_auth_credential_to_dict(cred):
    """Parse credential to dict"""

    def validate(cred):
        valid_keys = ['auth_type', 'auth']
        for k in valid_keys:
            if k not in cred:
                raise ValueError('Missing key in auth information, the correct format contains %s.' % valid_keys)
    try:
        _cred = json.loads(cred)
    except ValueError as e:
        LOG.error('Failed to parse credential with error: %s' % e)
        raise ValueError('Failed to parse credential, please check your Stack Credential format.')
    validate(_cred)
    return _cred