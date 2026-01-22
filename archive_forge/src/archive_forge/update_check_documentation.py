from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core.updater import schemas
from googlecloudsdk.core.util import files
import six
Notify the user of any available updates.

    This should be called for every command that is run.  It does not actually
    do an update check, and does not necessarily notify the user each time.  The
    user will only be notified if there are activated notifications and if the
    trigger for one of the activated notifications matches.  At most one
    notification will be printed per command.  Order or priority is determined
    by the order in which the notifications are registered in the component
    snapshot file.

    Args:
      command_path: str, The '.' separated path of the command that is currently
        being run (i.e. gcloud.foo.bar).
    