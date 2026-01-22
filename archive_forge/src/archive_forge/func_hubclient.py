from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.container.fleet import client
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
@property
def hubclient(self):
    """The HubClient for the current release track."""
    if not hasattr(self, '_client'):
        self._client = client.HubClient(self.ReleaseTrack())
    return self._client