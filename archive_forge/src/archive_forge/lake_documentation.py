from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.command_lib.iam import iam_util
Waits for the given google.longrunning.Operation to complete.