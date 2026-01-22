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