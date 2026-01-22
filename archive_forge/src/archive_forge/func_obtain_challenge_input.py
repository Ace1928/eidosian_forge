import abc
import base64
import sys
import pyu2f.convenience.authenticator
import pyu2f.errors
import pyu2f.model
import six
from google_reauth import _helpers, errors
def obtain_challenge_input(self, metadata):
    raise errors.ReauthSamlLoginRequiredError()