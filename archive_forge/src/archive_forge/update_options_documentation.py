from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataflow import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.dataflow import job_utils
Called when the user runs gcloud dataflow jobs update-options ...

    Args:
      args: all the arguments that were provided to this command invocation.

    Returns:
      The updated Job
    