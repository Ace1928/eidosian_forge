import abc
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
@classmethod
def validate_volume_add(cls, volume):
    """Validate that the volume dict has all needed parameters for this type."""
    required_keys = set(cls.required_fields().keys())
    optional_keys = set(cls.optional_fields().keys())
    for key in volume:
        if key == 'name':
            continue
        elif key == 'type':
            if volume[key] != cls.name():
                raise serverless_exceptions.ConfigurationError('expected volume of type {} but got {}'.format(cls.name(), volume[key]))
        elif key not in required_keys and key not in optional_keys:
            raise serverless_exceptions.ConfigurationError('Volume {} of type {} had unexpected parameter {}'.format(volume['name'], volume['type'], key))
    missing = required_keys - volume.keys()
    if missing:
        raise serverless_exceptions.ConfigurationError('Volume {} of type {} requires the following parameters: {}'.format(volume['name'], volume['type'], missing))