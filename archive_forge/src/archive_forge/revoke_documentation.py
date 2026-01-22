from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from google.oauth2 import credentials as google_auth_creds
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exc
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import google_auth_credentials as c_google_auth
from googlecloudsdk.core.credentials import store as c_store
Revoke Application Default Credentials.