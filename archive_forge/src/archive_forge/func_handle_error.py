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
def handle_error(self, error, url):
    """Try to resolve the given error, which should be a response
        to the given URL, by discharging any macaroon contained in
        it. That is, if error.code is ERR_DISCHARGE_REQUIRED
        then it will try to discharge err.info.macaroon. If the discharge
        succeeds, the discharged macaroon will be saved to the client's cookie
        jar, otherwise an exception will be raised.
        """
    if error.info is None or error.info.macaroon is None:
        raise BakeryException('unable to read info in discharge error response')
    discharges = bakery.discharge_all(error.info.macaroon, self.acquire_discharge, self.key)
    macaroons = '[' + ','.join(map(utils.macaroon_to_json_string, discharges)) + ']'
    all_macaroons = base64.urlsafe_b64encode(utils.to_bytes(macaroons))
    full_path = urljoin(url, error.info.macaroon_path)
    if error.info.cookie_name_suffix is not None:
        name = 'macaroon-' + error.info.cookie_name_suffix
    else:
        name = 'macaroon-auth'
    expires = checkers.macaroons_expiry_time(checkers.Namespace(), discharges)
    self.cookies.set_cookie(utils.cookie(name=name, value=all_macaroons.decode('ascii'), url=full_path, expires=expires))