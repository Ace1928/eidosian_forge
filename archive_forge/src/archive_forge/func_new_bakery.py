import base64
import datetime
import json
import platform
import threading
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import macaroonbakery.httpbakery as httpbakery
import pymacaroons
import requests
import macaroonbakery._utils as utils
from macaroonbakery.httpbakery._error import DischargeError
from fixtures import (
from httmock import HTTMock, urlmatch
from six.moves.urllib.parse import parse_qs
from six.moves.urllib.request import Request
def new_bakery(location, locator, checker):
    """Return a new bakery instance.
    @param location Location of the bakery {str}.
    @param locator Locator for third parties {ThirdPartyLocator or None}
    @param checker Caveat checker {FirstPartyCaveatChecker or None}
    @return {Bakery}
    """
    if checker is None:
        c = checkers.Checker()
        c.namespace().register('testns', '')
        c.register('is', 'testns', check_is_something)
        checker = c
    key = bakery.generate_key()
    return bakery.Bakery(location=location, locator=locator, key=key, checker=checker)