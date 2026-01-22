import datetime
import json
import logging
from oauth2client import _helpers
from oauth2client import client
from oauth2client import transport
from google_reauth import errors
from google_reauth import reauth
Refresh the access_token using the refresh_token.

        Args:
            http: An object to be used to make HTTP requests.
            rapt_refreshed: If we did or did not already refreshed the rapt
                            token.

        Raises:
            oauth2client.client.HttpAccessTokenRefreshError: if the refresh
                fails.
        