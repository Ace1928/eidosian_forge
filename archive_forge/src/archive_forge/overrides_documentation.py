from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.fleet import gkehub_api_util
from googlecloudsdk.api_lib.container.fleet.connectgateway import util as connectgateway_api_util
from googlecloudsdk.command_lib.container.fleet.memberships import util as memberships_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
Initializes the context manager.

    Args:
      region: The Connect Gateway region to connect to.

    Raises:
      exceptions.Error: If `region` is Falsy.
    