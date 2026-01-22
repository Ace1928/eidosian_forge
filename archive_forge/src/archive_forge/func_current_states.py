from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Dict
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet import util as fleet_util
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.command_lib.container.fleet.policycontroller import exceptions
from googlecloudsdk.core import exceptions as gcloud_exceptions
import six
def current_states(self) -> SpecMapping:
    """Fetches the current states from the server.

    If the feature is not enabled, this will return an empty dictionary.

    Returns:
      dictionary mapping from full path to membership spec.
    """
    try:
        return self.hubclient.ToPyDict(self.GetFeature().membershipStates)
    except gcloud_exceptions.Error as e:
        fne = self.FeatureNotEnabledError()
        if six.text_type(e) == six.text_type(fne):
            return dict()
        else:
            raise e