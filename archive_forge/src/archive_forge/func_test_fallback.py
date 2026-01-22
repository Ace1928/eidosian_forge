import unittest
import macaroonbakery.bakery as bakery
import macaroonbakery.httpbakery as httpbakery
from httmock import HTTMock, urlmatch
def test_fallback(self):
    key = bakery.generate_key()

    @urlmatch(path='.*/discharge/info')
    def discharge_info(url, request):
        return {'status_code': 404}

    @urlmatch(path='.*/publickey')
    def public_key(url, request):
        return {'status_code': 200, 'content': {'PublicKey': str(key.public_key)}}
    expectInfo = bakery.ThirdPartyInfo(public_key=key.public_key, version=bakery.VERSION_1)
    kr = httpbakery.ThirdPartyLocator(allow_insecure=True)
    with HTTMock(discharge_info):
        with HTTMock(public_key):
            info = kr.third_party_info('http://0.1.2.3/')
    self.assertEqual(info, expectInfo)