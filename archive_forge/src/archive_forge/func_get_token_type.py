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
def get_token_type(self, subject_token_type):
    return 'urn:ietf:params:oauth:token-type:jwt'