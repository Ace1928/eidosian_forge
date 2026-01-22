from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.tasks import GetApiAdapter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.tasks import flags
from googlecloudsdk.command_lib.tasks import parsers
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
Purge a queue by deleting all of its tasks.

  This command purges a queue by deleting all of its tasks. Purge operations can
  take up to one minute to take effect. Tasks might be dispatched before the
  purge takes effect. A purge is irreversible. All tasks created before this
  command is run are permanently deleted.
  