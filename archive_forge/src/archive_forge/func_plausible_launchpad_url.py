import base64
import re
from urllib.parse import urlparse, urlunparse
from ... import bedding, branch, errors, osutils, trace, transport
from ...i18n import gettext
from launchpadlib.credentials import (AccessToken, Credentials,
from launchpadlib.launchpad import Launchpad
@staticmethod
def plausible_launchpad_url(url):
    """Is 'url' something that could conceivably be pushed to LP?

        :param url: A URL that may refer to a Launchpad branch.
        :return: A boolean.
        """
    if url is None:
        return False
    if url.startswith('lp:'):
        return True
    regex = re.compile('([a-z]*\\+)*(bzr\\+ssh|http)://bazaar.*.launchpad.net')
    return bool(regex.match(url))