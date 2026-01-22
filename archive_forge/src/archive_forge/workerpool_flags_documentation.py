from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
Set up all the argparse flags for updating a workerpool.

  Args:
    parser: An argparse.ArgumentParser-like object.
    release_track: A base.ReleaseTrack-like object.

  Returns:
    The parser argument with workerpool flags added in.
  