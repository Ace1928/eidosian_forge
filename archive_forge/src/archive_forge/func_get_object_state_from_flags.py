from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def get_object_state_from_flags(flag_args):
    """Returns object version to query based on user flags."""
    if getattr(flag_args, 'soft_deleted', False):
        return cloud_api.ObjectState.SOFT_DELETED
    if getattr(flag_args, 'all_versions', False):
        return cloud_api.ObjectState.LIVE_AND_NONCURRENT
    return cloud_api.ObjectState.LIVE