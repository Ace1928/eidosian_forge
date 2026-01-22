from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import json
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
@property
def oauth_token_url(self):
    api = 'v1/oauthtoken'
    return '{}/{}'.format(self._base_url, api)