import base64
import json
from collections import namedtuple
from datetime import timedelta
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import pymacaroons
from macaroonbakery.tests.common import epoch, test_checker, test_context
from pymacaroons.verifier import FirstPartyCaveatVerifierDelegate, Verifier
def do(self, ctx, svc, ops):

    class _AuthInfo:
        authInfo = None

    def svc_do(ms):
        _AuthInfo.authInfo = svc.do(ctx, ms, ops)
    self._do_func(ctx, svc, svc_do)
    return _AuthInfo.authInfo