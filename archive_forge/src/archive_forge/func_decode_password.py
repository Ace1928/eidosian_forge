from ... import version_info  # noqa: F401
from ... import config, errors, lazy_import
from ... import transport as _mod_transport
def decode_password(self, credentials):
    auth = self._netrc.authenticators(credentials['host'])
    password = None
    if auth is not None:
        user, account, password = auth
        cred_user = credentials.get('user', None)
        if cred_user is None or user != cred_user:
            password = None
    return password