import time
from boto.provider import Provider
def has_google_credentials():
    global _HAS_GOOGLE_CREDENTIALS
    if _HAS_GOOGLE_CREDENTIALS is None:
        provider = Provider('google')
        if provider.get_access_key() is None or provider.get_secret_key() is None:
            _HAS_GOOGLE_CREDENTIALS = False
        else:
            _HAS_GOOGLE_CREDENTIALS = True
    return _HAS_GOOGLE_CREDENTIALS