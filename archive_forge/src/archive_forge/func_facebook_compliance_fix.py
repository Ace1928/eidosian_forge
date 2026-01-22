from json import dumps
from urllib.parse import parse_qsl
def facebook_compliance_fix(session):

    def _compliance_fix(r):
        if 'application/json' in r.headers.get('content-type', {}):
            return r
        if 'text/plain' in r.headers.get('content-type', {}) and r.status_code == 200:
            token = dict(parse_qsl(r.text, keep_blank_values=True))
        else:
            return r
        expires = token.get('expires')
        if expires is not None:
            token['expires_in'] = expires
        token['token_type'] = 'Bearer'
        r._content = dumps(token).encode()
        return r
    session.register_compliance_hook('access_token_response', _compliance_fix)
    return session