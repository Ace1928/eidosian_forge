import copy
import json
import logging
from collections import namedtuple
import macaroonbakery.bakery as bakery
import macaroonbakery.httpbakery as httpbakery
import macaroonbakery._utils as utils
import requests.cookies
from six.moves.urllib.parse import urljoin
@property
def login_url(self):
    """ Return the URL from which to acquire a macaroon that can be used
        to complete the agent login. To acquire the macaroon, make a POST
        request to the URL with user and public-key parameters.
        :return string
        """
    return self._login_url