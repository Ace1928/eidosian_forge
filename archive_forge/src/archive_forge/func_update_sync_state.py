from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet.config_management import utils
from googlecloudsdk.command_lib.container.fleet.features import base as feature_base
from googlecloudsdk.core import log
def update_sync_state(self, fs):
    """Update config_sync state for the membership that has ACM installed.

    Args:
      fs: ConfigManagementFeatureState
    """
    if not (fs.configSyncState and fs.configSyncState.syncState):
        self.config_sync = 'SYNC_STATE_UNSPECIFIED'
    else:
        self.config_sync = fs.configSyncState.syncState.code
        if fs.configSyncState.syncState.syncToken:
            self.last_synced_token = fs.configSyncState.syncState.syncToken[:7]
        self.last_synced = fs.configSyncState.syncState.lastSyncTime
        if has_config_sync_git(fs):
            self.sync_branch = fs.membershipSpec.configSync.git.syncBranch