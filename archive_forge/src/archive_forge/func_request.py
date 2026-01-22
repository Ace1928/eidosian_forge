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
def request(self, method, url, **kwargs):
    """Use the requests library to make a request.
        Using this method is like doing:

            requests.request(method, url, auth=client.auth())
        """
    kwargs['auth'] = self.auth()
    kwargs['cookies'] = self.cookies
    return requests.request(method=method, url=url, **kwargs)