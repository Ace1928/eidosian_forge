from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet.config_management import utils
from googlecloudsdk.command_lib.container.fleet.features import base as feature_base
from googlecloudsdk.core import log
def config_sync_state(state):
    """Convert state to a string shown to the users.

  Args:
    state: a string from the ACM Fleet Feature state representing the Config
    Sync state.

  Returns:
    a string shown to the users representing the Config Sync state.
  """
    if state == 'CONFIG_SYNC_INSTALLED':
        return 'INSTALLED'
    elif state == 'CONFIG_SYNC_NOT_INSTALLED':
        return 'NOT_INSTALLED'
    elif state == 'CONFIG_SYNC_ERROR':
        return 'ERROR'
    elif state == 'CONFIG_SYNC_PENDING':
        return 'PENDING'
    else:
        return 'UNSPECIFIED'