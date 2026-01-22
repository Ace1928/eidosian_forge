import codecs
import types
import six
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
def describe_field(field_definition):
    """Build descriptor for Field instance.

    Args:
      field_definition: Field instance to provide descriptor for.

    Returns:
      Initialized FieldDescriptor instance describing the Field instance.
    """
    field_descriptor = FieldDescriptor()
    field_descriptor.name = field_definition.name
    field_descriptor.number = field_definition.number
    field_descriptor.variant = field_definition.variant
    if isinstance(field_definition, messages.EnumField):
        field_descriptor.type_name = field_definition.type.definition_name()
    if isinstance(field_definition, messages.MessageField):
        field_descriptor.type_name = field_definition.message_type.definition_name()
    if field_definition.default is not None:
        field_descriptor.default_value = _DEFAULT_TO_STRING_MAP[type(field_definition)](field_definition.default)
    if field_definition.repeated:
        field_descriptor.label = FieldDescriptor.Label.REPEATED
    elif field_definition.required:
        field_descriptor.label = FieldDescriptor.Label.REQUIRED
    else:
        field_descriptor.label = FieldDescriptor.Label.OPTIONAL
    return field_descriptor