import copy
import json
import logging
from collections import namedtuple
import macaroonbakery.bakery as bakery
import macaroonbakery.httpbakery as httpbakery
import macaroonbakery._utils as utils
import requests.cookies
from six.moves.urllib.parse import urljoin
def load_auth_info(filename):
    """Loads agent authentication information from the specified file.
    The returned information is suitable for passing as an argument
    to the AgentInteractor constructor.
    @param filename The name of the file to open (str)
    @return AuthInfo The authentication information
    @raises AgentFileFormatError when the file format is bad.
    """
    with open(filename) as f:
        return read_auth_info(f.read())