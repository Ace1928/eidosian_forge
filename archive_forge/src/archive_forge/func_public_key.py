import unittest
import macaroonbakery.bakery as bakery
import macaroonbakery.httpbakery as httpbakery
from httmock import HTTMock, urlmatch
@urlmatch(path='.*/publickey')
def public_key(url, request):
    return {'status_code': 200, 'content': {'PublicKey': str(key.public_key)}}