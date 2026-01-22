import base64
import re
from urllib.parse import urlparse, urlunparse
from ... import bedding, branch, errors, osutils, trace, transport
from ...i18n import gettext
from launchpadlib.credentials import (AccessToken, Credentials,
from launchpadlib.launchpad import Launchpad
def do_save(self, credentials, unique_key):
    """Store newly-authorized credentials in the keyring."""
    self.auth_config._set_option(unique_key, 'consumer_key', credentials.consumer.key)
    self.auth_config._set_option(unique_key, 'consumer_secret', credentials.consumer.secret)
    self.auth_config._set_option(unique_key, 'access_token', credentials.access_token.key)
    self.auth_config._set_option(unique_key, 'access_secret', credentials.access_token.secret)