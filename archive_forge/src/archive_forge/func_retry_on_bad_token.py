import errno
import os
import warnings
from lazr.restfulclient.resource import (  # noqa: F401
from lazr.restfulclient.authorize.oauth import SystemWideConsumer
from lazr.restfulclient._browser import RestfulHttp
from launchpadlib.credentials import (
from launchpadlib import uris
from launchpadlib.uris import (  # noqa: F401
def retry_on_bad_token(self, response, content, *args):
    """If the response indicates a bad token, get a new token and retry.

        Otherwise, just return the response.
        """
    if self._bad_oauth_token(response, content) and self.authorization_engine is not None:
        self.launchpad.credentials.access_token = None
        self.authorization_engine(self.launchpad.credentials, self.launchpad.credential_store)
        return self._request(*args)
    return (response, content)