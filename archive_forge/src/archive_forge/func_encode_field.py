import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def encode_field(self, field, value):
    """Encode the given value as JSON.

        Args:
          field: a messages.Field for the field we're encoding.
          value: a value for field.

        Returns:
          A python value suitable for json.dumps.
        """
    for encoder in _GetFieldCodecs(field, 'encoder'):
        result = encoder(field, value)
        value = result.value
        if result.complete:
            return value
    if isinstance(field, messages.EnumField):
        if field.repeated:
            remapped_value = [GetCustomJsonEnumMapping(field.type, python_name=e.name) or e.name for e in value]
        else:
            remapped_value = GetCustomJsonEnumMapping(field.type, python_name=value.name)
        if remapped_value:
            return remapped_value
    if isinstance(field, messages.MessageField) and (not isinstance(field, message_types.DateTimeField)):
        value = json.loads(self.encode_message(value))
    return super(_ProtoJsonApiTools, self).encode_field(field, value)