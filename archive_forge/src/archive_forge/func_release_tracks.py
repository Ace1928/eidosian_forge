import abc
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
@classmethod
def release_tracks(cls):
    return [base.ReleaseTrack.ALPHA]