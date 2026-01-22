import base64
import json
import logging
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import macaroonbakery._utils as utils
from ._browser import WebBrowserInteractor
from ._error import (
from ._interactor import (
import requests
from six.moves.http_cookies import SimpleCookie
from six.moves.urllib.parse import urljoin
def extract_macaroons(headers_or_request):
    """ Returns an array of any macaroons found in the given slice of cookies.
    If the argument implements a get_header method, that will be used
    instead of the get method to retrieve headers.
    @param headers_or_request: dict of headers or a
    urllib.request.Request-like object.
    @return: A list of list of mpy macaroons
    """

    def get_header(key, default=None):
        try:
            return headers_or_request.get_header(key, default)
        except AttributeError:
            return headers_or_request.get(key, default)
    mss = []

    def add_macaroon(data):
        try:
            data = utils.b64decode(data)
            data_as_objs = json.loads(data.decode('utf-8'))
        except ValueError:
            return
        ms = [utils.macaroon_from_dict(x) for x in data_as_objs]
        mss.append(ms)
    cookie_header = get_header('Cookie')
    if cookie_header is not None:
        cs = SimpleCookie()
        cs.load(str(cookie_header))
        for c in cs:
            if c.startswith('macaroon-'):
                add_macaroon(cs[c].value)
    macaroon_header = get_header('Macaroons')
    if macaroon_header is not None:
        for h in macaroon_header.split(','):
            add_macaroon(h)
    return mss