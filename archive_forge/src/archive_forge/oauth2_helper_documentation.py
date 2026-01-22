from __future__ import absolute_import
import io
import json
import os
import sys
import time
import webbrowser
from gcs_oauth2_boto_plugin import oauth2_client
import oauth2client.client
from six.moves import input  # pylint: disable=redefined-builtin
Run the OAuth2 flow to fetch a refresh token. Returns the refresh token.