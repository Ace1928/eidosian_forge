from requests import auth
from requests import cookies
from . import _digest_auth_compat as auth_compat, http_proxy_digest
def handle_407(self, r, **kwargs):
    proxy_authenticate = r.headers.get('Proxy-Authenticate', '').lower()
    if 'basic' in proxy_authenticate:
        return self._handle_basic_auth_407(r, kwargs)
    if 'digest' in proxy_authenticate:
        return self._handle_digest_auth_407(r, kwargs)