from contextlib import contextmanager
import launchpadlib
from launchpadlib.launchpad import Launchpad
from launchpadlib.credentials import (
def make_end_user_authorize_token(self, credentials, request_token):
    """Pretend to exchange a request token for an access token.

        We do this by simply setting the access_token property.
        """
    credentials.access_token = AccessToken(self.ACCESS_TOKEN_KEY, 'access_secret:168')
    self.access_tokens_obtained += 1