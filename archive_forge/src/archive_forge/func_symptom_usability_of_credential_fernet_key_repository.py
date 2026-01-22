import keystone.conf
from keystone.common import fernet_utils as utils
from keystone.credential.providers import fernet as credential_fernet
def symptom_usability_of_credential_fernet_key_repository():
    """Credential key repository is not setup correctly.

    The credential Fernet key repository is expected to be readable by the user
    running keystone, but not world-readable, because it contains
    security sensitive secrets.
    """
    fernet_utils = utils.FernetUtils(CONF.credential.key_repository, credential_fernet.MAX_ACTIVE_KEYS, 'credential')
    return 'fernet' in CONF.credential.provider and (not fernet_utils.validate_key_repository())