import abc
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
@classmethod
def fill_volume(cls, volume, new_vol, messages):
    src = messages.SecretVolumeSource(secretName=volume['secret'])
    item = messages.KeyToPath(path=volume['path'], key=volume['version'])
    src.items.append(item)
    new_vol.secret = src