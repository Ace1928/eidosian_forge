import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import macaroonbakery.httpbakery as httpbakery
import macaroonbakery.httpbakery.agent as agent
import requests.cookies
from httmock import HTTMock, response, urlmatch
from six.moves.urllib.parse import parse_qs, urlparse
def test_agent_login(self):
    discharge_key = bakery.generate_key()

    class _DischargerLocator(bakery.ThirdPartyLocator):

        def third_party_info(self, loc):
            if loc == 'http://0.3.2.1':
                return bakery.ThirdPartyInfo(public_key=discharge_key.public_key, version=bakery.LATEST_VERSION)
    d = _DischargerLocator()
    server_key = bakery.generate_key()
    server_bakery = bakery.Bakery(key=server_key, locator=d)

    @urlmatch(path='.*/here')
    def server_get(url, request):
        ctx = checkers.AuthContext()
        test_ops = [bakery.Op(entity='test-op', action='read')]
        auth_checker = server_bakery.checker.auth(httpbakery.extract_macaroons(request.headers))
        try:
            auth_checker.allow(ctx, test_ops)
            resp = response(status_code=200, content='done')
        except bakery.PermissionDenied:
            caveats = [checkers.Caveat(location='http://0.3.2.1', condition='is-ok')]
            m = server_bakery.oven.macaroon(version=bakery.LATEST_VERSION, expiry=datetime.utcnow() + timedelta(days=1), caveats=caveats, ops=test_ops)
            content, headers = httpbakery.discharge_required_response(m, '/', 'test', 'message')
            resp = response(status_code=401, content=content, headers=headers)
        return request.hooks['response'][0](resp)

    @urlmatch(path='.*/discharge')
    def discharge(url, request):
        qs = parse_qs(request.body)
        if qs.get('token64') is None:
            return response(status_code=401, content={'Code': httpbakery.ERR_INTERACTION_REQUIRED, 'Message': 'interaction required', 'Info': {'InteractionMethods': {'agent': {'login-url': '/login'}}}}, headers={'Content-Type': 'application/json'})
        else:
            qs = parse_qs(request.body)
            content = {q: qs[q][0] for q in qs}
            m = httpbakery.discharge(checkers.AuthContext(), content, discharge_key, None, alwaysOK3rd)
            return {'status_code': 200, 'content': {'Macaroon': m.to_dict()}}
    auth_info = agent.load_auth_info(self.agent_filename)

    @urlmatch(path='.*/login')
    def login(url, request):
        qs = parse_qs(urlparse(request.url).query)
        self.assertEqual(request.method, 'GET')
        self.assertEqual(qs, {'username': ['test-user'], 'public-key': [PUBLIC_KEY]})
        b = bakery.Bakery(key=discharge_key)
        m = b.oven.macaroon(version=bakery.LATEST_VERSION, expiry=datetime.utcnow() + timedelta(days=1), caveats=[bakery.local_third_party_caveat(PUBLIC_KEY, version=httpbakery.request_version(request.headers))], ops=[bakery.Op(entity='agent', action='login')])
        return {'status_code': 200, 'content': {'macaroon': m.to_dict()}}
    with HTTMock(server_get), HTTMock(discharge), HTTMock(login):
        client = httpbakery.Client(interaction_methods=[agent.AgentInteractor(auth_info)])
        resp = requests.get('http://0.1.2.3/here', cookies=client.cookies, auth=client.auth())
    self.assertEqual(resp.content, b'done')