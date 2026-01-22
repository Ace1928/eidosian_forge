from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import tempfile
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import download_util
from googlecloudsdk.command_lib.artifacts import file_util
from googlecloudsdk.command_lib.artifacts import flags
from googlecloudsdk.core import log
def downloadGenericArtifact(self, args, repo_ref, file_id, file_name):
    final_path = os.path.join(args.destination, file_name)
    if args.name:
        tmp_path = os.path.join(tempfile.gettempdir(), file_name)
    else:
        tmp_path = final_path
    file_escaped = file_util.EscapeFileNameFromIDs(repo_ref.projectsId, repo_ref.locationsId, repo_ref.repositoriesId, file_id)
    download_util.Download(tmp_path, final_path, file_escaped.RelativeName(), False)
    log.status.Print('Successfully downloaded the file to {}'.format(args.destination))