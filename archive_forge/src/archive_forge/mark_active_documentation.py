from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.recommender import recommendation
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.recommender import flags
Run 'gcloud recommender recommendations mark-active'.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Returns:
      The result recommendations after being marked as active
    